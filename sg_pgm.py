import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.ops import point_to_node_partition, index_select
from modules.registration import get_node_correspondences
from modules.sinkhorn import LearnableLogOptimalTransport
from modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointRescoreMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from modules.backbone import KPConvFPN

from modules.layers.affinity import Affinity_Double, Affinity
from modules.layers.gnn import GATEncoder
from modules.k_pred.afau import Encoder
from modules.k_pred.sinkhorn_topk import soft_topk, greedy_perm
from modules.layers.sinkhorn import sinkhorn_rpm
from modules.ops.hungarian import hungarian
from modules.pointnet.pointnet import PointNetfeat

from torch_geometric.nn import  max_pool_x, GATv2Conv, knn_graph
from torch_geometric.utils import to_dense_batch
#from utils.pointcloud import *


class SGPGM_GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(SGPGM_GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.model_name = "SGM_GeoTr"

        self.use_ptfusion = cfg.model_variant.pt_fusion
        self.use_sgfusion = cfg.model_variant.sg_fusion
        self.use_topk = cfg.model_variant.soft_topk

        if self.use_ptfusion:
            self.model_name += "_ptfuse"
        if self.use_sgfusion:
            self.model_name += "_sgfuse" # It has no learnale parameters, anyway...
        if self.use_topk:
            self.model_name += "_topk"

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )
        if self.use_sgfusion:
            # Super Point Matching Rescoring
            self.coarse_matching = SuperPointRescoreMatching(
                cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
            )
        else:
            self.coarse_matching = SuperPointMatching(
                cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
            )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=None,#cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        # Scene Graph Network
        self.sgfeat_net = GATEncoder(cfg.scene_graph_net.input_dim, recep_field=cfg.scene_graph_net.num_layers, 
                                    node_input_dim=cfg.scene_graph_net.node_input_dim, edge_input_dim=cfg.scene_graph_net.edge_input_dim)
        if self.use_ptfusion:
            self.sg_affinity = Affinity_Double(d1=cfg.scene_graph_net.input_dim*(cfg.scene_graph_net.num_layers+1), d2=cfg.backbone.output_dim)
            self.sg_emb_out_dim = cfg.scene_graph_net.input_dim*(cfg.scene_graph_net.num_layers+1) + cfg.backbone.output_dim
        else:
            self.sg_affinity = Affinity(d=cfg.scene_graph_net.input_dim*(cfg.scene_graph_net.num_layers+1))
            self.sg_emb_out_dim = cfg.scene_graph_net.input_dim*(cfg.scene_graph_net.num_layers+1)
        self.sg_inst_norm = nn.InstanceNorm2d(1, affine=True)

        self.sinkhorn_iters = cfg.scene_graph_net.sinkhorn_iters
        self.sinkhorn_epsilon = cfg.scene_graph_net.sinkhorn_epsilon

        if self.use_ptfusion:
            self.pt2sg_gnn = GATv2Conv(in_channels=cfg.backbone.output_dim, out_channels=cfg.backbone.output_dim)

        if self.use_topk:
            self.univ_size = cfg.AFA.UNIV_SIZE
            self.k_top_encoder = Encoder(cfg)
            self.maxpool = nn.MaxPool1d(kernel_size=self.univ_size)
            self.final_row = nn.Sequential(
                        nn.Linear(self.univ_size, 8),
                        nn.ReLU(),
                        nn.Linear(8, 1),
                        nn.Sigmoid()
                    )
            self.final_col = nn.Sequential(
                        nn.Linear(self.univ_size, 8),
                        nn.ReLU(),
                        nn.Linear(8, 1),
                        nn.Sigmoid()
                    )

    def forward(self, data_dict, early_stop=True):
        output_dict = {}
        self.batch_size = data_dict['batch_size']

        # 0. Unpack Inputs
        feats = data_dict['features'].detach()
        points = data_dict['points'][0].detach()
        transform = data_dict['transform'].detach()

        ref_length = data_dict['lengths'][0][0].item() 
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length_c = data_dict['lengths'][-1][0].item()
        
        ref_points_f, src_points_f = data_dict['points'][1][:ref_length_f].detach(), data_dict['points'][1][ref_length_f:].detach()
        ref_points_c, src_points_c = data_dict['points'][-1][:ref_length_c].detach(), data_dict['points'][-1][ref_length_c:].detach()

        ref_insts_f, src_insts_f = data_dict['insts'][1][:ref_length_f].detach(), data_dict['insts'][1][ref_length_f:].detach()
        ref_insts_c, src_insts_c = data_dict['insts'][-1][:ref_length_c].detach(), data_dict['insts'][-1][ref_length_c:].detach()

        # 1. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict) # fine 256, middle 512, coarse 1024
        ref_feats_f, src_feats_f = feats_list[0][:ref_length_f], feats_list[0][ref_length_f:]
        ref_feats_c, src_feats_c =  feats_list[-1][:ref_length_c], feats_list[-1][ref_length_c:]

        # 2. Scene Graph Matching
        sg_pair = data_dict['sg']
        sg_emb_q = self.sgfeat_net(sg_pair.x_q, sg_pair.edge_index_q, sg_pair.edge_attr_q)
        sg_emb_t = self.sgfeat_net(sg_pair.x_t, sg_pair.edge_index_t, sg_pair.edge_attr_t)
        sg_emb_q_batched, sg_q_mask = to_dense_batch(sg_emb_q, sg_pair.x_q_batch, fill_value=0.0)
        sg_emb_t_batched, sg_t_mask = to_dense_batch(sg_emb_t, sg_pair.x_t_batch, fill_value=0.0)

        _, ns_q = torch.unique(sg_pair.x_q_batch, return_counts=True)
        _, ns_t = torch.unique(sg_pair.x_t_batch, return_counts=True)

        if self.use_ptfusion:
            pt2sg_emb_q = self.ptEmb_to_sgEmb(ref_feats_f, ref_insts_f, sg_pair.x_q_batch, ref_points_f)
            pt2sg_emb_t = self.ptEmb_to_sgEmb(src_feats_f, src_insts_f, sg_pair.x_t_batch, src_points_f)
            sg_emb_q_batched, pt2sg_emb_q = F.normalize(sg_emb_q_batched, p=2, dim=1), F.normalize(pt2sg_emb_q, p=2, dim=1)
            sg_emb_t_batched, pt2sg_emb_t = F.normalize(sg_emb_t_batched, p=2, dim=1), F.normalize(pt2sg_emb_t, p=2, dim=1)
            output_dict['ref_ptsg'] = pt2sg_emb_q
            output_dict['src_ptsg'] = pt2sg_emb_t
        
        output_dict['sg_feat_q'] = sg_emb_q_batched
        output_dict['sg_feat_t'] = sg_emb_t_batched
        
        sg_emb_q_fused = torch.cat((sg_emb_q_batched, pt2sg_emb_q), dim=-1) if self.use_ptfusion else sg_emb_q_batched
        sg_emb_t_fused = torch.cat((sg_emb_t_batched, pt2sg_emb_t), dim=-1) if self.use_ptfusion else sg_emb_t_batched
        sg_log_alpha = self.sg_affinity(sg_emb_q_fused, sg_emb_t_fused)
        sg_log_alpha = self.sg_inst_norm(sg_log_alpha[:,None,:,:]).squeeze(dim=1)

        sg_matches_padded = sinkhorn_rpm(log_alpha=sg_log_alpha, n_iters=self.sinkhorn_iters, eps=self.sinkhorn_epsilon)
        sg_matches = sg_matches_padded[:,:-1,:-1]
        sg_matches_raw = sg_matches.clone()
        output_dict['sg_matches_raw'] = sg_matches_raw
        
        # Top-K Partial Matching
        if self.use_topk:
            dummy_row = self.univ_size - sg_matches.shape[1]
            dummy_col = self.univ_size - sg_matches.shape[2]

            init_row_emb = torch.zeros((sg_matches.shape[0], sg_matches.shape[1],self.univ_size), dtype=torch.float32, device=sg_matches.device)
            init_col_emb = torch.zeros((sg_matches.shape[0], sg_matches.shape[2],self.univ_size), dtype=torch.float32, device=sg_matches.device)
            for b in range(sg_matches.shape[0]):
                index = torch.linspace(0, sg_matches.shape[2]-1, sg_matches.shape[2], dtype=torch.long, device=sg_matches.device).unsqueeze(1)
                init_col_emb_one = torch.zeros(sg_matches.shape[2], self.univ_size, dtype=torch.float32, device=sg_matches.device).scatter_(1, index, 1)
                init_col_emb[b] = init_col_emb_one

            out_emb_row, out_emb_col = self.k_top_encoder(init_row_emb, init_col_emb, sg_matches.detach())
            out_emb_row = torch.nn.functional.pad(out_emb_row, (0, 0, 0, dummy_row),value=float('-inf')).permute(0, 2, 1)
            out_emb_col = torch.nn.functional.pad(out_emb_col, (0, 0, 0, dummy_col),value=float('-inf')).permute(0, 2, 1)

            global_row_emb = self.maxpool(out_emb_row).squeeze(-1)
            global_col_emb = self.maxpool(out_emb_col).squeeze(-1)
            k_row = self.final_row(global_row_emb).squeeze(-1)
            k_col = self.final_col(global_col_emb).squeeze(-1)
            ks = (k_row + k_col) / 2
            
            output_dict['top_k'] = ks

            if self.training:
                ks_gt = torch.tensor([data_dict['sg_match'][i].shape[0] for i in range(len(data_dict['sg_match']))], dtype=torch.long, device=sg_matches.device)
                _, sg_matches = soft_topk(sg_matches, ks_gt, self.sinkhorn_iters, 0.05, ns_q, ns_t,True)
            else:
                _, sg_matches = soft_topk(sg_matches, ks*torch.minimum(ns_q, ns_t), self.sinkhorn_iters, 0.05, ns_q, ns_t,True)

            output_dict['sg_matches'] = sg_matches
            x = hungarian(sg_matches, ns_q, ns_t)
            top_indices = torch.argsort(x.mul(sg_matches).reshape(x.shape[0], -1), descending=True, dim=-1)
            x = torch.zeros(sg_matches.shape, device=sg_matches.device)
            x = greedy_perm(x, top_indices, ks.view(-1) * torch.minimum(ns_q, ns_t))
            sg_matches_sparse = torch.nonzero(x.squeeze())
        else:
            deny_thr=5e-2
            output_dict['sg_matches'] = sg_matches
            sg_matches_detach = sg_matches.clone().detach().squeeze()
            sg_matches_detach[sg_matches_detach < deny_thr] = 0.
            sg_matches_sparse = torch.vstack([torch.arange(sg_matches_detach.shape[0], device='cuda'), torch.argmax(sg_matches_detach, dim=-1)])
            zero_mask = sg_matches_detach[sg_matches_sparse.T[:, 0], sg_matches_sparse.T[:, 1]] > 0
            sg_matches_sparse = sg_matches_sparse[:,zero_mask].T

        output_dict['sg_matches_sparse'] = sg_matches_sparse

        if early_stop: # Since we only train for graph matching, we can stop here.
            output_dict['ref_feats_f'] = ref_feats_f
            output_dict['src_feats_f'] = src_feats_f
            output_dict['ref_insts_f'] = ref_insts_f
            output_dict['src_insts_f'] = src_insts_f
            return output_dict
        
        # 2.5 SG match filtering
        if self.training:
            sg_match_gt = data_dict['sg_match'][0]
        else:
            sg_match_gt = sg_matches_sparse
        
        for i in range(self.batch_size):
            filter_mask = (ref_insts_c == sg_match_gt[:,0]).any(dim=-1)
            ref_feats_c, ref_points_c, ref_insts_c = ref_feats_c[filter_mask], ref_points_c[filter_mask], ref_insts_c[filter_mask]
            filter_mask = (src_insts_c == sg_match_gt[:,1]).any(dim=-1)
            src_feats_c, src_points_c, src_insts_c = src_feats_c[filter_mask], src_points_c[filter_mask], src_insts_c[filter_mask]
            
            filter_mask = (ref_insts_f == sg_match_gt[:,0]).any(dim=-1)
            ref_feats_f, ref_points_f, ref_insts_f = ref_feats_f[filter_mask], ref_points_f[filter_mask], ref_insts_f[filter_mask]
            filter_mask = (src_insts_f == sg_match_gt[:,1]).any(dim=-1)
            src_feats_f, src_points_f, src_insts_f = src_feats_f[filter_mask], src_points_f[filter_mask], src_insts_f[filter_mask]
        
        # 3. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        ref_padded_insts_f = torch.cat([ref_insts_f, torch.zeros_like(ref_insts_f[:1])], dim=0)
        src_padded_insts_f = torch.cat([src_insts_f, torch.zeros_like(src_insts_f[:1])], dim=0)
        ref_node_knn_insts = index_select(ref_padded_insts_f, ref_node_knn_indices, dim=0)
        src_node_knn_insts = index_select(src_padded_insts_f, src_node_knn_indices, dim=0)

        if self.training:
            matching_radius = data_dict['voxel_size']*2
            gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
                ref_points_c,
                src_points_c,
                ref_node_knn_points,
                src_node_knn_points,
                transform,
                matching_radius,
                ref_masks=ref_node_masks,
                src_masks=src_node_masks,
                ref_knn_masks=ref_node_knn_masks,
                src_knn_masks=src_node_knn_masks,
            )

            output_dict['gt_node_corr_indices'] = gt_node_corr_indices
            output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps
            if len(output_dict['gt_node_corr_indices']) == 0:
                return output_dict
            
            if ref_points_c.shape[0] <= 3 or src_points_c.shape[0] <= 3:
                output_dict['gt_node_corr_indices'] = []
                return output_dict

        # 4. Conditional Transformer
        ref_feats_c, src_feats_c, _ = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )

        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)
        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            if self.use_sgfusion:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                    ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks, 
                    ref_insts_c, src_insts_c, sg_matches_raw, ks
                )
            else:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                    ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
                )
            
            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_node_corr_knn_insts = ref_node_knn_insts[ref_node_corr_indices]  # (P, K, 1)
        src_node_corr_knn_insts = src_node_knn_insts[src_node_corr_indices]  # (P, K, 1)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / (src_feats_f.shape[1]+ref_feats_f.shape[1]) ** 0.5#feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)
        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform, three_indices = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            batch_indices, ref_indices, src_indices = three_indices
            ref_corr_insts = ref_node_corr_knn_insts[batch_indices, ref_indices]
            src_corr_insts = src_node_corr_knn_insts[batch_indices, src_indices]
            
            output_dict['ref_corr_insts'] = ref_corr_insts
            output_dict['src_corr_insts'] = src_corr_insts

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform
        
        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_insts_f'] = ref_insts_f
        output_dict['src_insts_f'] = src_insts_f
        output_dict['ref_points'] = points[:ref_length]
        output_dict['src_points'] = points[ref_length:]

        return output_dict
    
    def ptEmb_to_sgEmb(self, pt_emb_, inst_map, sg_batch, pt_c):
        pooled_pt_emb = []
        pt_emb = pt_emb_.detach().clone()
        _, ns_sg = torch.unique(sg_batch, return_counts=True)
        
        inst_map = inst_map.squeeze()
        idle_batch = torch.zeros(inst_map.shape[0], device=pt_emb_.device)
        pooled_pt_idxs = torch.unique(inst_map).long()

        pt_edge = knn_graph(pt_c, k=3)
        pt_emb = self.pt2sg_gnn(pt_emb, pt_edge)

        pooled_pt_emb = torch.zeros((ns_sg, pt_emb.shape[-1]), device=pt_emb_.device)
        pooled_pt_emb[pooled_pt_idxs,:] = max_pool_x(inst_map, pt_emb, idle_batch)[0]

        pooled_pt_emb_batched, _  = to_dense_batch(pooled_pt_emb, sg_batch, fill_value=0.0)
        return pooled_pt_emb_batched

    def save_weights(self, root: str, epoch: int, iteration: int):
        all_state_dict = self.state_dict()
        sg_pgm_state_dict = {}
        for key in all_state_dict.keys():
            if 'backbone' not in key and 'transformer' not in key and 'optimal_transport' not in key:
                sg_pgm_state_dict[key] = all_state_dict[key]
        path = os.path.join(root, self.model_name + '_' + str(epoch) + '_' + str(iteration) + '.pth')
        torch.save(sg_pgm_state_dict, path)
    
    def load_weights(self, path_sgm: str, path_geo,report: bool = False):
        state_dict_sgm = torch.load(path_sgm)
        state_dict_geo = torch.load(path_geo)['model']
        model_dict = self.state_dict()
        model_dict.update(state_dict_sgm)
        model_dict.update(state_dict_geo)
        self.load_state_dict(model_dict)
        if report:
            for name, param in model_dict.items():
                    print(name, param.shape)
    
    def clean_weights(self, path: str, report: bool = False): #TODO: remove this function
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        test_dict = {}
        for key in state_dict.keys():
            if 'backbone' not in key and 'transformer' not in key and 'optimal_transport' not in key:
                test_dict[key] = state_dict[key]
        torch.save(test_dict, 'best_model.pth')
        self.load_state_dict(state_dict)
        if report:
            for name, param in state_dict.items():
                print(name, param.shape)

    def load_pretrained(self, path:str, report: bool = False):
        state_dict_geo = torch.load(path)['model']
        model_dict = self.state_dict()
        model_dict.update(state_dict_geo)
        self.load_state_dict(model_dict)
        if report:
            for name, param in model_dict.items():
                    print(name, param.shape)
    
    def lock_geotr(self):
        self.backbone.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.optimal_transport.requires_grad_(False)


class SGPGM_PointNet(nn.Module):
    def __init__(self, cfg, use_ptfusion=False, use_topk=True):
        super(SGPGM_PointNet, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.model_name = "SGM_PointNet"

        self.use_ptfusion = use_ptfusion
        self.use_topk = use_topk

        if self.use_ptfusion:
            self.model_name += "_ptfuse"
        if self.use_topk:
            self.model_name += "_topk"

        self.object_encoder = PointNetfeat(global_feat=True, batch_norm=True, point_size=3, input_transform=False, feature_transform=False, out_size=200)

        # Scene Graph Network
        self.sgfeat_net = GATEncoder(cfg.scene_graph_net.input_dim, recep_field=cfg.scene_graph_net.num_layers, 
                                    node_input_dim=cfg.scene_graph_net.node_input_dim, edge_input_dim=cfg.scene_graph_net.edge_input_dim)
        if self.use_ptfusion:
            self.sg_affinity = Affinity_Double(d1=cfg.scene_graph_net.input_dim*(cfg.scene_graph_net.num_layers+1), d2=200)
            self.sg_emb_out_dim = cfg.scene_graph_net.input_dim*(cfg.scene_graph_net.num_layers+1) + 200
        else:
            self.sg_affinity = Affinity(d=cfg.scene_graph_net.input_dim*(cfg.scene_graph_net.num_layers+1))
            self.sg_emb_out_dim = cfg.scene_graph_net.input_dim*(cfg.scene_graph_net.num_layers+1)
        self.sg_inst_norm = nn.InstanceNorm2d(1, affine=True)

        self.sinkhorn_iters = cfg.scene_graph_net.sinkhorn_iters
        self.sinkhorn_epsilon = cfg.scene_graph_net.sinkhorn_epsilon

        if self.use_ptfusion:
            self.pt2sg_gnn = GATv2Conv(in_channels=cfg.backbone.output_dim, out_channels=cfg.backbone.output_dim)

        if self.use_topk:
            self.univ_size = cfg.AFA.UNIV_SIZE
            self.k_top_encoder = Encoder(cfg)
            self.maxpool = nn.MaxPool1d(kernel_size=self.univ_size)
            self.final_row = nn.Sequential(
                        nn.Linear(self.univ_size, 8),
                        nn.ReLU(),
                        nn.Linear(8, 1),
                        nn.Sigmoid()
                    )
            self.final_col = nn.Sequential(
                        nn.Linear(self.univ_size, 8),
                        nn.ReLU(),
                        nn.Linear(8, 1),
                        nn.Sigmoid()
                    )

    def forward(self, data_dict):
        output_dict = {}
        self.batch_size = data_dict['batch_size']

        # 1. Scene Graph Matching
        sg_pair = data_dict['sg']
        sg_emb_q = self.sgfeat_net(sg_pair.x_q, sg_pair.edge_index_q, sg_pair.edge_attr_q)
        sg_emb_t = self.sgfeat_net(sg_pair.x_t, sg_pair.edge_index_t, sg_pair.edge_attr_t)
        sg_emb_q_batched, sg_q_mask = to_dense_batch(sg_emb_q, sg_pair.x_q_batch, fill_value=0.0)
        sg_emb_t_batched, sg_t_mask = to_dense_batch(sg_emb_t, sg_pair.x_t_batch, fill_value=0.0)

        _, ns_q = torch.unique(sg_pair.x_q_batch, return_counts=True)
        _, ns_t = torch.unique(sg_pair.x_t_batch, return_counts=True)

        ref_pts = data_dict['ref_points']
        src_pts = data_dict['src_points']
        pt_feats = self.object_encoder(torch.cat([ref_pts, src_pts]).permute(0, 2, 1)) # Nx200
        pt2sg_emb_q = pt_feats[:ref_pts.shape[0]].unsqueeze(dim=0)
        pt2sg_emb_t = pt_feats[ref_pts.shape[0]:].unsqueeze(dim=0)
        
        if self.use_ptfusion:
            sg_emb_q_batched, pt2sg_emb_q = F.normalize(sg_emb_q_batched, p=2, dim=1), F.normalize(pt2sg_emb_q, p=2, dim=1)
            sg_emb_t_batched, pt2sg_emb_t = F.normalize(sg_emb_t_batched, p=2, dim=1), F.normalize(pt2sg_emb_t, p=2, dim=1)
            output_dict['ref_ptsg'] = pt2sg_emb_q
            output_dict['src_ptsg'] = pt2sg_emb_t
        
        output_dict['sg_feat_q'] = sg_emb_q_batched
        output_dict['sg_feat_t'] = sg_emb_t_batched
        
        sg_emb_q_fused = torch.cat((sg_emb_q_batched, pt2sg_emb_q), dim=-1) if self.use_ptfusion else sg_emb_q_batched
        sg_emb_t_fused = torch.cat((sg_emb_t_batched, pt2sg_emb_t), dim=-1) if self.use_ptfusion else sg_emb_t_batched
        sg_log_alpha = self.sg_affinity(sg_emb_q_fused, sg_emb_t_fused)
        sg_log_alpha = self.sg_inst_norm(sg_log_alpha[:,None,:,:]).squeeze(dim=1)

        sg_matches_padded = sinkhorn_rpm(log_alpha=sg_log_alpha, n_iters=self.sinkhorn_iters, eps=self.sinkhorn_epsilon)
        sg_matches = sg_matches_padded[:,:-1,:-1]
        sg_matches_raw = sg_matches.clone()
        output_dict['sg_matches_raw'] = sg_matches_raw
        
        if self.use_topk:
            dummy_row = self.univ_size - sg_matches.shape[1]
            dummy_col = self.univ_size - sg_matches.shape[2]

            init_row_emb = torch.zeros((sg_matches.shape[0], sg_matches.shape[1],self.univ_size), dtype=torch.float32, device=sg_matches.device)
            init_col_emb = torch.zeros((sg_matches.shape[0], sg_matches.shape[2],self.univ_size), dtype=torch.float32, device=sg_matches.device)
            for b in range(sg_matches.shape[0]):
                index = torch.linspace(0, sg_matches.shape[2]-1, sg_matches.shape[2], dtype=torch.long, device=sg_matches.device).unsqueeze(1)
                init_col_emb_one = torch.zeros(sg_matches.shape[2], self.univ_size, dtype=torch.float32, device=sg_matches.device).scatter_(1, index, 1)
                init_col_emb[b] = init_col_emb_one

            out_emb_row, out_emb_col = self.k_top_encoder(init_row_emb, init_col_emb, sg_matches.detach())
            out_emb_row = torch.nn.functional.pad(out_emb_row, (0, 0, 0, dummy_row),value=float('-inf')).permute(0, 2, 1)
            out_emb_col = torch.nn.functional.pad(out_emb_col, (0, 0, 0, dummy_col),value=float('-inf')).permute(0, 2, 1)

            global_row_emb = self.maxpool(out_emb_row).squeeze(-1)
            global_col_emb = self.maxpool(out_emb_col).squeeze(-1)
            k_row = self.final_row(global_row_emb).squeeze(-1)
            k_col = self.final_col(global_col_emb).squeeze(-1)
            ks = (k_row + k_col) / 2
            output_dict['top_k'] = ks

            if self.training:
                ks_gt = torch.tensor([data_dict['sg_match'][i].shape[0] for i in range(len(data_dict['sg_match']))], dtype=torch.long, device=sg_matches.device)
                _, sg_matches = soft_topk(sg_matches, ks_gt, self.sinkhorn_iters, 0.05, ns_q, ns_t,True)
            else:
                _, sg_matches = soft_topk(sg_matches, ks*torch.minimum(ns_q, ns_t), self.sinkhorn_iters, 0.05, ns_q, ns_t,True)

            output_dict['sg_matches'] = sg_matches
            x = hungarian(sg_matches, ns_q, ns_t)
            top_indices = torch.argsort(x.mul(sg_matches).reshape(x.shape[0], -1), descending=True, dim=-1)
            x = torch.zeros(sg_matches.shape, device=sg_matches.device)
            x = greedy_perm(x, top_indices, ks.view(-1) * torch.minimum(ns_q, ns_t))
            sg_matches_sparse = torch.nonzero(x.squeeze())
        else:
            deny_thr=5e-2
            output_dict['sg_matches'] = sg_matches
            sg_matches_detach = sg_matches.clone().detach().squeeze()
            sg_matches_detach[sg_matches_detach < deny_thr] = 0.
            sg_matches_sparse = torch.vstack([torch.arange(sg_matches_detach.shape[0], device='cuda'), torch.argmax(sg_matches_detach, dim=-1)])
            zero_mask = sg_matches_detach[sg_matches_sparse.T[:, 0], sg_matches_sparse.T[:, 1]] > 0
            sg_matches_sparse = sg_matches_sparse[:,zero_mask].T

        output_dict['sg_matches_sparse'] = sg_matches_sparse

        return output_dict
    
    
    def ptEmb_to_sgEmb(self, pt_emb_, inst_map, sg_batch, pt_c):
        pooled_pt_emb = []
        pt_emb = pt_emb_.detach().clone()
        _, ns_sg = torch.unique(sg_batch, return_counts=True)
        
        inst_map = inst_map.squeeze()
        idle_batch = torch.zeros(inst_map.shape[0], device=pt_emb_.device)
        pooled_pt_idxs = torch.unique(inst_map).long()

        pt_edge = knn_graph(pt_c, k=3)
        pt_emb = self.pt2sg_gnn(pt_emb, pt_edge)

        pooled_pt_emb = torch.zeros((ns_sg, pt_emb.shape[-1]), device=pt_emb_.device)
        pooled_pt_emb[pooled_pt_idxs,:] = max_pool_x(inst_map, pt_emb, idle_batch)[0]

        pooled_pt_emb_batched, _  = to_dense_batch(pooled_pt_emb, sg_batch, fill_value=0.0)
        return pooled_pt_emb_batched

    def save_weights(self, root: str, epoch: int, iteration: int):
        path = os.path.join(root, self.model_name + '_' + str(epoch) + '_' + str(iteration) + '.pth')
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path: str, report: bool = False):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        if report:
            for name, param in state_dict.items():
                print(name, param.shape)