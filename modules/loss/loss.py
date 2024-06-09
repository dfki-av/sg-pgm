import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.pointcloud import compute_registration_error, get_nearest_neighbor
from utils.pointcloud import apply_transform as apply_transform_np
#from utils.registration import compute_registration_error, compute_registration_rmse

class SceneGraphMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(SceneGraphMatchingLoss, self).__init__()

    def forward(self, permute_pred, permute_gt, eps=1e-8):
        loss = []
        B = permute_pred.shape[0]
        for i in range(B):

            permute_gt_per_sample = permute_gt[i].long()
            val = permute_pred[i][permute_gt_per_sample[:,0], permute_gt_per_sample[:,1]]
            term = val + eps
            loss.append(torch.mean(-torch.log(term)))
        return torch.mean(torch.stack(loss))
class SceneGraphKPredictionLoss(nn.Module):
    def __init__(self):
        super(SceneGraphKPredictionLoss, self).__init__()
    def forward(self, output_dict, data_dict):
        k_loss = 0.
        if 'top_k' in output_dict.keys():
            ks = output_dict['top_k']
            supervised_ks = float(data_dict['sg_match'][0].shape[0] / min(output_dict['sg_matches'].shape[1], output_dict['sg_matches'].shape[2]))
            supervised_ks = torch.tensor([supervised_ks], device=ks.device)
            k_loss = F.mse_loss(ks, supervised_ks)
        return k_loss

class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.sg_loss = SceneGraphMatchingLoss(cfg)
        self.weight_topk_loss = cfg.loss.weight_topk_loss


    def forward(self, output_dict, data_dict, topk_stage=True, sg_stage=True):
        sg_loss = self.sg_loss(output_dict['sg_matches'], data_dict['sg_match'])
        k_loss = 0.
        if 'top_k' in output_dict.keys():
            ks = output_dict['top_k']
            supervised_ks = float(data_dict['sg_match'][0].shape[0] / min(output_dict['sg_matches'].shape[1], output_dict['sg_matches'].shape[2]))
            supervised_ks = torch.tensor([supervised_ks], device=ks.device)
            k_loss = F.mse_loss(ks, supervised_ks)
            output_dict['ks_gt'] = supervised_ks
        
        loss = sg_loss + self.weight_topk_loss*k_loss

        return {
            'loss': loss,
            's_l': sg_loss,
            'k_l': self.weight_topk_loss*k_loss
        }


class Evaluator(nn.Module):
    def __init__(self, cfg, eval_geo=True, eval_sgm=True, eval_coarse=True):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold
        self.inlier_ratio_thresh = 0.05
        self.eval_geo = eval_geo
        self.eval_sgm = eval_sgm
        self.eval_coarse = eval_coarse
        self.eval_types = ['Overlap', 'IR', 'CCD', 'RRE', 'RTE', 'FMR', 'RMSE', 'RR', 'CorrS', 'HIT1', 'HIT3', 'HIT5', 'MRR', 'F1']


    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = np.greater(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = np.zeros([ref_length_c, src_length_c])
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform_np(src_corr_points, transform)
        corr_distances = np.linalg.norm(ref_corr_points - src_corr_points, axis=1)
        precision = np.less(corr_distances, self.acceptance_radius).astype(np.float32).mean()
        fmr = (precision >= self.inlier_ratio_thresh).astype(np.float32)
        return precision, fmr

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']
        ref_points = output_dict['ref_points']
        
        rre, rte = compute_registration_error(transform, est_transform)

        realignment_transform = np.matmul(np.linalg.inv(transform), est_transform)
        realigned_src_points_f = apply_transform_np(src_points, realignment_transform)
        rmse = np.linalg.norm(realigned_src_points_f - src_points, axis=1).mean()
        recall = np.less(rmse, self.acceptance_rmse).astype(np.float32)


        raw_points = data_dict['raw_points']
        ccd = self.compute_modified_chamfer_distance(src_points, ref_points, raw_points, est_transform, transform)

        return rre, rte, rmse, recall, ccd

    @torch.no_grad()
    def evaluate_sgmatch(self, output_dict, data_dict):
        match_gt_sparse = data_dict['sg_match']
        match_pred = output_dict['sg_matches']
        match_pred_sparse = output_dict['sg_matches_sparse']

        hit1 = self.compute_hits_k(match_pred, match_gt_sparse, k=1)
        hit3 = self.compute_hits_k(match_pred, match_gt_sparse, k=3)
        hit5 = self.compute_hits_k(match_pred, match_gt_sparse, k=5)

        mrr = self.compute_mean_reciprocal_rank(match_pred, match_gt_sparse)

        match_gt_dense = np.zeros_like(match_pred)
        
        for i in range(match_pred.shape[0]): 
            match_gt_dense[i][match_gt_sparse[i][:,0],  match_gt_sparse[i][:,1]]  = 1.

        match_pred_recover = np.zeros_like(match_pred)
        for i in range(match_pred.shape[0]): 
            match_pred_recover[i][match_pred_sparse[:,0],  match_pred_sparse[:,1]]  = 1.
        tp,fp,fn = self.get_pos_neg(match_pred_recover, match_gt_dense)

        const = 1e-7
        precision = tp / (tp + fp + const)
        recall = tp / (tp + fn + const)
        f1 = 2 * precision * recall / (precision + recall + const)

        #CorrS
        corrS = 0
        if self.eval_geo:
            ref_corr_insts = output_dict['ref_corr_insts']
            src_corr_insts = output_dict['src_corr_insts']
            corrS = match_gt_dense[0,:,:][ref_corr_insts.squeeze(), src_corr_insts.squeeze()].sum() / ref_corr_insts.shape[0]

        return  hit1, hit3, hit5, mrr, f1, corrS

    @torch.no_grad()
    def get_pos_neg(self, pmat_pred, pmat_gt):
        """
        Calculates number of true positives, false positives and false negatives
        :param pmat_pred: predicted permutation matrix
        :param pmat_gt: ground truth permutation matrix
        :return: tp, fp, fn
        """
        #device = pmat_pred.device
        #pmat_gt = pmat_gt.to(device)

        tp = np.sum(pmat_pred * pmat_gt).astype(np.float32)
        fp = np.sum(pmat_pred * (1 - pmat_gt)).astype(np.float32)
        fn = np.sum((1 - pmat_pred) * pmat_gt).astype(np.float32)
        return tp, fp, fn
    
    @torch.no_grad()
    def hit_at_k(self, permute_pred, permute_gt_sparse, k=1, reduction='mean', deny_thr=1e-3):
        assert reduction in ['mean', 'sum']
        B = permute_pred.shape[0]
        #permute_pred[permute_pred < deny_thr] = 0.
        terms = 0.
        for idx in range(B):
            permute_gt_per_batch = permute_gt_sparse[idx]
            permute_pred_sorted, indices = torch.sort(permute_pred[idx], dim=-1, descending=True)
            k = min(k, indices.shape[-1])
            hit_k = indices[:,:k]
            correct = 0
            for i in range(k):
                correct +=  (hit_k[permute_gt_per_batch[:, 0], i] == permute_gt_per_batch[:, 1]).sum()
            terms += (correct / permute_gt_per_batch.shape[0]) if reduction == 'mean' else correct
        
        return terms / B if reduction == 'mean' else terms
    
    @torch.no_grad()
    def compute_hits_k(self, permute_pred, permute_gt_sparse, k=1, deny_thr=1e-3):
        B = permute_pred.shape[0]
        for idx in range(B):
            permute_gt_per_batch = permute_gt_sparse[idx]
            e1i_idxs, e2i_idxs = permute_gt_per_batch[:,1], permute_gt_per_batch[:,0]
            rank_list = np.argsort(-permute_pred[idx], axis=-1)
            correct, total = 0, 0

            for i, e2i_idx in enumerate(e2i_idxs):
                e2_idx_rank_list = list(rank_list[e2i_idx])
                e2_idx_rank_list_k = e2_idx_rank_list[:k]
                if e1i_idxs[i] in e2_idx_rank_list_k: correct += 1
            
            total = e2i_idxs.shape[0]
            if total > 0:
                return float(correct)/float(total)
            else: return 0
    
    @torch.no_grad()
    def compute_mean_reciprocal_rank(self, permute_pred, permute_gt_sparse):
        B = permute_pred.shape[0]
        mrr_arr = 0

        for idx in range(B):
            permute_gt_per_batch = permute_gt_sparse[idx]
            if permute_gt_per_batch.shape[0] > 0:
                e2i_idxs, e1i_idxs = permute_gt_per_batch[:,0], permute_gt_per_batch[:,1]
                rank_list = np.argsort(-permute_pred[idx], axis=-1)
                e2_idx_rank_list = rank_list[e2i_idxs,:]
                _, rs = np.nonzero(e2_idx_rank_list == np.tile(e1i_idxs[:, np.newaxis], (1, e2_idx_rank_list.shape[1])))#[:,1]
                mrr_arr = mrr_arr + np.mean(1. / (rs + 1.))
            else:
                mrr_arr = mrr_arr + 0
            
        return mrr_arr/B

    @torch.no_grad()
    def forward(self, output_dict, data_dict):
        
        eval_metrics_dict = {key:0 for key in self.eval_types}

        if self.eval_geo:
            if self.eval_coarse:
                eval_metrics_dict['PIR'] = self.evaluate_coarse(output_dict)
            eval_metrics_dict['IR'], eval_metrics_dict['FMR'] = self.evaluate_fine(output_dict, data_dict)
            eval_metrics_dict['RRE'], eval_metrics_dict['RTE'], eval_metrics_dict['RMSE'], eval_metrics_dict['RR'], eval_metrics_dict['CCD'] = \
                self.evaluate_registration(output_dict, data_dict)

        if self.eval_sgm:
            eval_metrics_dict['HIT1'], eval_metrics_dict['HIT3'], eval_metrics_dict['HIT5'], \
            eval_metrics_dict['MRR'],  eval_metrics_dict['F1'], eval_metrics_dict['CorrS'] = self.evaluate_sgmatch(output_dict, data_dict)
            
        eval_metrics_dict['Overlap'] = data_dict['overlap']
        return eval_metrics_dict

    @torch.no_grad()
    def compute_modified_chamfer_distance(self, src_points, ref_points, raw_points, est_transform, gt_transform):
        aligned_src_points = apply_transform_np(src_points, est_transform)
        chamfer_distance_p_q = get_nearest_neighbor(aligned_src_points, raw_points).mean()
        composed_transform = np.matmul(est_transform, np.linalg.inv(gt_transform))
        aligned_raw_points = apply_transform_np(raw_points, composed_transform)
        chamfer_distance_q_p = get_nearest_neighbor(ref_points, aligned_raw_points).mean()

        chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p
        return chamfer_distance
