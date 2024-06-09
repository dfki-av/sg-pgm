import os
import datetime, time
import argparse
import torch
import torch.utils.data


from modules.net_geo import GeoTransformer_pure
from data.config import Config
from data.datasets import get_datasets
from data.stack_mode import get_dataloader, registration_collate_fn_stack_mode
import data.torch_utils as torch_utils 

from utils.util import MovingAverage
from utils.logger import SGMatchLogger
from utils.set_seed import set_reproducibility
from utils.pointcloud import *
from data.torch_utils import to_cuda, release_cuda
from modules.loss.loss import OverallLoss, Evaluator

import pygcransac

difficult_list = [4, 9, 14, 27, 29, 56, 92, 123, 138, 152, 172, 240, 267, 271, 274, 329, 335, 361, 365, 379, 380, 
                  398, 410, 414, 429, 442, 453, 498, 542, 543, 561, 584, 590, 624, 631, 672, 683, 707, 727, 749, 762, 
                  764, 769, 794, 805, 828, 844, 845, 850, 871, 889, 931, 937, 942, 943, 948, 949, 962, 963, 969, 970, 
                  1004, 1045, 1051, 1055, 1059, 1113, 1119, 1145, 1149, 1174, 1185, 1191, 1267, 1283, 1305, 1307, 1309, 
                  1311, 1329, 1377, 1388, 1391, 1428, 1438, 1452, 1462, 1483, 1494, 1522, 1550, 1551, 1570, 1579, 1583, 
                  1590, 1591, 1618, 1626, 1642, 1644, 1650, 1655, 1656, 1675, 1702, 1704, 1707, 1714, 1721, 1752, 1779, 
                  1784, 1796, 1810, 1831, 1847]


eval_types = ['IR', 'FMR', 'RRE', 'RTE', 'RMSE', 'RR']

def parse_args(argv=None):
    parser  = argparse.ArgumentParser(description='3D Scene Graph Macthing Testing')
    parser.add_argument('--mode',  choices=['train', 'infer', 'eval'], default='train', help='' )
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
   
    # Basic Settings
    parser.add_argument('--config', type=str, default="./data/sg_pgm.json", help='Configuration to be loaded.')
    parser.add_argument('--dataset_root', type=str, default="/home/xie/Documents/datasets/3RScan", help='Path for load dataset.')
    parser.add_argument('--dataset', choices=['sgm', 'sgm_tar', 'sgm_pointnet'], default='sgm', help='Path for load dataset.')
    parser.add_argument('--num_workers', default=16, type=int, help='')
    parser.add_argument('--split',  choices=['train', 'valid', 'all'], default='all', help='' )
    parser.add_argument('--device',  choices=['cuda', 'cpu'], default='cuda', help='Device to train on.' )
    parser.add_argument('--reproducibility', default=True, action='store_true', help='Set true if you want to reproduct the almost same results as given in the ablation study.') 

    # Training Settings
    parser.add_argument('--eval_num', default=200, type=int, help='Sample numbers to use for evaluation.')
    parser.add_argument('--save_folder', type=str, default="./weights", help='Path for saving training weights.')
    parser.add_argument('--log_root', type=str, default='./logs', help='Path for saving training logs.')

    # Inference Settings
    parser.add_argument('--trained_model', type=str, default=None, help='Path of the trained weights.')
    
    global args
    args = parser.parse_args(argv)

    global cfg
    assert os.path.exists(args.config), print("Configuration doesn't exist!")
    cfg = Config(args.config)

    if args.reproducibility:
        set_reproducibility()
    
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

def inference(cfg, eval_num=-1, min_object_points=50, num_p2p_corrs=20000, vis=False):
    # Initialize Network, Loss and Optimizer
    model = GeoTransformer_pure(cfg).cuda()
    model.load_weights("./weights/geotransformer-3dmatch.pth.tar")

    ransac_threshold = 0.03
    ransac_min_iters = 5000
    ransac_max_iters = 5000
    ransac_use_sprt = False
    inlier_ratio_thresh = 0.05
    rmse_thresh = 0.2

    criterion = OverallLoss(cfg=cfg)
    evaluator = Evaluator(cfg=cfg, eval_sgm=False, eval_coarse=False)
    print("model loaded.")

    train_dataset, val_dataset = get_datasets(args)
    print("Dataset loaded.")

    # Initialize accumulator and timer
    frame_times = MovingAverage()
    
    who_breaked = []
    # Start Training
    model.eval()
    if eval_num < 0: eval_num = val_dataset.__len__()

    result_dict_avgs = {k: [] for k in eval_types}
    result_dict_avgs_simp = {k: [] for k in eval_types}
    result_dict_avgs_ovlp = {k: [] for k in eval_types}
    
    #for idx, sample in enumerate(val_dataset):
    for idx in difficult_list:
        sample = val_dataset.__getitem__(idx)
        src_points = sample['src_points']
        ref_points = sample['ref_points']
        src_inst = sample['src_insts']
        ref_inst = sample['ref_insts']
        node_corrs = sample['sg_match']
        gt_transform = sample['transform']

        '''# Test on GeoTr with all point cloud
        output_dict_simp = perform_registration(model, src_points, ref_points, src_inst, ref_inst, gt_transform)
        result_dict_simp = evaluator(output_dict_simp, sample)
        for k in result_dict_avgs_simp:
            result_dict_avgs_simp[k].append(result_dict_simp[k])
        

        # Test on GeoTr with ovelapped point cloud
        filter_mask = np.isin(ref_inst, node_corrs[:,0])
        ref_inst_o, ref_points_o = ref_inst[filter_mask], ref_points[filter_mask]
        filter_mask =  np.isin(src_inst, node_corrs[:,1])
        src_inst_o, src_points_o = src_inst[filter_mask], src_points[filter_mask]
        
        
        pcd1 = make_open3d_point_cloud(ref_points_o)
        pcd2 = make_open3d_point_cloud(src_points_o)
        pcd1.estimate_normals()
        pcd2.estimate_normals()
        draw_registration_result(pcd2, pcd1, gt_transform, recolor=True)
        

        output_dict_ovlp = perform_registration(model, src_points_o, ref_points_o, src_inst_o, ref_inst_o, gt_transform)
        result_dict_ovlp = evaluator(output_dict_ovlp, sample)
        for k in result_dict_avgs_ovlp:
            result_dict_avgs_ovlp[k].append(result_dict_ovlp[k])'''
        
        #print(sample["overlap"])
        '''pcd1 = make_open3d_point_cloud(ref_points)
        pcd2 = make_open3d_point_cloud(src_points)
        pcd1.estimate_normals()
        pcd2.estimate_normals()
        draw_registration_result(pcd2, pcd1, est_transform, recolor=True)'''

        # Test on GeoTr with object to object
        point_corrs = {'src' : [], 'ref' : [], 'scores' : []}
        for node_corr in node_corrs:
            node_points_src = src_points[src_inst  == node_corr[1].item(),:]
            node_points_ref = ref_points[ref_inst  == node_corr[0].item(),:]
            node_inst_src = np.ones([node_points_src.shape[0], 1])*node_corr[0].item()
            node_inst_ref = np.ones([node_points_ref.shape[0], 1])*node_corr[1].item()
            if node_points_src.shape[0] < min_object_points or node_points_ref.shape[0] < min_object_points: continue
            output_dict = perform_registration(model, node_points_src, node_points_ref, node_inst_src, node_inst_ref, gt_transform)
            if output_dict is None: continue

            output_dict = torch_utils.release_cuda(output_dict)
            ref_corr_points = output_dict['ref_corr_points']
            src_corr_points = output_dict['src_corr_points']
            corr_scores = output_dict['corr_scores']

            if corr_scores.shape[0] > num_p2p_corrs // len(node_corrs):
                sel_indices = np.argsort(-corr_scores)[: num_p2p_corrs // len(node_corrs)]
                ref_corr_points = ref_corr_points[sel_indices]
                src_corr_points = src_corr_points[sel_indices]

            point_corrs['src'].append(src_corr_points)
            point_corrs['ref'].append(ref_corr_points)
            point_corrs['scores'].append(corr_scores)

            pred = {'ref_corr_points': ref_corr_points, 'src_corr_points' :
             src_corr_points,'estimated_transform': output_dict['estimated_transform']
            ,'src_points': node_points_src,'ref_points':node_points_ref}

            result_dict_per_obj = evaluator(pred, {'transform': gt_transform})
            print()
            print('   '.join('{}: {:5.4f}'.format(k, result_dict_per_obj[k]) for k in result_dict_per_obj))
            pcd1 = make_open3d_point_cloud(node_points_ref)
            pcd2 = make_open3d_point_cloud(node_points_src)
            pcd1.estimate_normals()
            pcd2.estimate_normals()
            draw_registration_result(pcd2, pcd1, gt_transform, recolor=True)
            draw_registration_result(pcd2, pcd1, output_dict['estimated_transform'], recolor=True)
            

        
        if len(point_corrs['src']) == 0 or len(point_corrs['ref']) == 0: return None
        
        point_corrs['src'] = np.concatenate(point_corrs['src'])
        point_corrs['ref'] = np.concatenate(point_corrs['ref'])

        corrs_ransac = np.concatenate([point_corrs['src'], point_corrs['ref']], axis=1)
        
        if corrs_ransac.shape[0] > num_p2p_corrs:
            corr_sel_indices = np.random.choice(corrs_ransac.shape[0], num_p2p_corrs)
            corrs_ransac = corrs_ransac[corr_sel_indices]

        est_transform, _ = pygcransac.findRigidTransform(np.ascontiguousarray(corrs_ransac), probabilities = [], 
                                                         threshold = ransac_threshold, neighborhood_size = 4, sampler = 1, 
                                                         min_iters = ransac_min_iters, max_iters = ransac_max_iters, 
                                                         spatial_coherence_weight = 0.0, 
                                                         use_space_partitioning = not ransac_use_sprt, neighborhood = 0, conf = 0.999, 
                                                         use_sprt = ransac_use_sprt)
        if est_transform is not None:
            est_transform = est_transform.T
        
            out_dict = {}
            out_dict['ref_corr_points'] = point_corrs['ref']
            out_dict['src_corr_points'] = point_corrs['src']
            out_dict['estimated_transform'] = est_transform
            out_dict['src_points'] = src_points
            out_dict['ref_points'] = ref_points

            pcd1 = make_open3d_point_cloud(ref_points)
            pcd2 = make_open3d_point_cloud(src_points)
            pcd1.estimate_normals()
            pcd2.estimate_normals()
            draw_registration_result(pcd2, pcd1, est_transform, recolor=True)
            draw_registration_result(pcd2, pcd1, gt_transform, recolor=True)
            

            data_dict={}
            data_dict['transform'] = gt_transform
            result_dict = evaluator(out_dict, data_dict)
            if result_dict['RRE'] > 3.0:
                who_breaked.append(idx)
            for k in result_dict_avgs:
                result_dict_avgs[k].append(result_dict[k])
            if idx >= eval_num:
                break

        print("\rEvaluation on going ... {:3.2f}% ".format((idx+1)/eval_num*100) + '   '.join('{}: {:5.4f}'.format(k, np.asarray(result_dict_avgs[k]).mean()) for k in result_dict_avgs), end='')

    model.train()
    for k in result_dict_avgs:
        result_dict_avgs[k] = np.asarray(result_dict_avgs[k]).mean()
    print()
    print("Evaluation Metrics Object to Object:")
    print('   '.join('{}: {:5.4f}'.format(k, result_dict_avgs[k]) for k in result_dict_avgs))
    
    for k in result_dict_avgs_simp:
        result_dict_avgs_simp[k] = np.asarray(result_dict_avgs_simp[k]).mean()
    print()
    print("Evaluation Metrics Pts_to_Pts:")
    print('   '.join('{}: {:5.4f}'.format(k, result_dict_avgs_simp[k]) for k in result_dict_avgs_simp))


    result_dict_avgs_ovlp
    for k in result_dict_avgs_ovlp:
        result_dict_avgs_ovlp[k] = np.asarray(result_dict_avgs_ovlp[k]).mean()
    print()
    print("Evaluation Metrics Ovelapped:")
    print('   '.join('{}: {:5.4f}'.format(k, result_dict_avgs_ovlp[k]) for k in result_dict_avgs_ovlp))

    print(who_breaked)
    return

def perform_registration(model, src_points, ref_points, inst_src, inst_ref, gt_transform):
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        'ref_insts': inst_ref.astype(np.float32),
        'src_insts': inst_src.astype(np.float32),
        "transform" : gt_transform.astype(np.float32)
    }
    
    '''pcd1 = make_open3d_point_cloud(ref_points)
    pcd2 = make_open3d_point_cloud(src_points)
    pcd1.estimate_normals()
    pcd2.estimate_normals()
    draw_registration_result(pcd2, pcd1, gt_transform, recolor=True)'''
    

    with torch.no_grad():
        neighbor_limits =cfg.dataset.neighbor_limits

        data_dict = registration_collate_fn_stack_mode([data_dict], 
                        cfg.backbone.num_stages, cfg.backbone.init_voxel_size, 
                        cfg.backbone.init_radius, neighbor_limits)
        # output dict
        data_dict = to_cuda(data_dict)
        try:
            output_dict = model(data_dict)
        except:
            return None
        
    output_dict = torch_utils.release_cuda(output_dict)

    return output_dict


if __name__ == '__main__':

    parse_args()
    inference(cfg=cfg, eval_num=-1)