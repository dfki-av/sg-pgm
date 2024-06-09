import os
import datetime, time
import argparse
import torch
import torch.utils.data
import pygcransac

from net import GeoTransformer
from data.config import Config
from data.datasets import get_datasets, PairData
from data.stack_mode import get_dataloader, registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode, build_dataloader_stack_mode
import data.torch_utils as torch_utils 
from datapipes.base_pipeline import BasePipelineCreator

from utils.util import MovingAverage
from utils.logger import SGMatchLogger
from utils.set_seed import set_reproducibility
from utils.pointcloud import *
from data.torch_utils import to_cuda, release_cuda
from modules.loss.loss import OverallLoss, Evaluator
from utils import cuda_timer
from utils import common
from utils.util_label import NYU40_Label_Names

EVAL_TYPES = ['Overlap', 'IR', 'CCD', 'RRE', 'RTE', 'FMR', 'RMSE', 'RR', 'CorrS',
              'HIT1', 'HIT3', 'HIT5', 'MRR', 'F1']

ransac_threshold = 0.03
ransac_min_iters = 5000
ransac_max_iters = 5000
ransac_use_sprt =True

def parse_args(argv=None):
    parser  = argparse.ArgumentParser(description='3D Scene Graph Macthing Testing')
    # Hyper Parameters for Training
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')

    # Basic Settings
    parser.add_argument('--config', type=str, default="./data/sg_pgm.json", help='Configuration to be loaded.')
    parser.add_argument('--dataset_root', type=str, default="/home/xie/Documents/datasets/3RScan/dp_fast", help='Path for load dataset.')
    parser.add_argument('--dataset', choices=['sgm', 'sgm_tar', 'sgm_pointnet'], default='sgm_tar', help='Path for load dataset.')
    parser.add_argument('--num_workers', default=12, type=int, help='')
    parser.add_argument('--split',  choices=['train', 'valid', 'all'], default='all', help='' )
    parser.add_argument('--device',  choices=['cuda', 'cpu'], default='cuda', help='Device to train on.' )
    parser.add_argument('--reproducibility', default=True, action='store_true', help='Set true if you want to reproduct the almost same results as given in the ablation study.') 

    # Training Settings
    parser.add_argument('--eval_num', default=-1, type=int, help='Sample numbers to use for evaluation.')
    parser.add_argument('--rand_trans', default=False, action='store_true', help='augmented random transformation between src and ref fragments.')
    parser.add_argument('--save_folder', type=str, default="./weights", help='Path for saving training weights.')
    parser.add_argument('--pretrained_3dmatch', type=str, default="./weights/geotransformer-3dmatch.pth.tar", help='Path for pretrained weights on 3dmatch.')

    # Inference Settings
    parser.add_argument('--trained_model', type=str, default=None, help='Path of the trained weights.')
    parser.add_argument('--seed', type=int, default=1000, help='define data sampling num for coarse points.')
    parser.add_argument('--from_json', type=str, default=None, help='from log file')
    # Net Variants
    parser.add_argument('--sgfusion', default=False, action='store_true', help='no sg fusion')
    parser.add_argument('--ptfusion', default=False, action='store_true', help='no pt fusion')
    parser.add_argument('--topk', default=False, action='store_true', help='no topK')
    parser.add_argument('--max_c_points', type=int, default=1500, help='define data sampling num for coarse points.')
    global args
    args = parser.parse_args(argv)

    global cfg
    assert os.path.exists(args.config), print("Configuration doesn't exist!")
    cfg = Config(args.config)

    if args.reproducibility:
        set_reproducibility(args.seed)
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

def evaluation(model, val_loader, evaluator: Evaluator, 
               logger: SGMatchLogger=None, epoch=None, eval_num=-1, eval_types=EVAL_TYPES):
    model.eval()
    
    if eval_num < 0: eval_num = val_loader.__len__()
    result_dict_avgs = {k: [] for k in eval_types}
    #pred_len = []
    
    with torch.no_grad():
        for idx, batched_data in enumerate(val_loader):
            if idx >= eval_num: break

            batched_data = to_cuda(batched_data)
            output_dict = model(batched_data, early_stop=False)

            batched_data = torch_utils.release_cuda(batched_data)
            output_dict = torch_utils.release_cuda(output_dict)

            ref_corr_points = output_dict['ref_corr_points']
            src_corr_points = output_dict['src_corr_points']
            corrs_ransac = np.concatenate([ref_corr_points, src_corr_points], axis=1)
            
            '''est_transform, _ = pygcransac.findRigidTransform(np.ascontiguousarray(corrs_ransac), probabilities = [], 
                                                        threshold = ransac_threshold, neighborhood_size = 4, sampler = 1, 
                                                        min_iters = ransac_min_iters, max_iters = ransac_max_iters, 
                                                        spatial_coherence_weight = 0.0, 
                                                        use_space_partitioning = not ransac_use_sprt, neighborhood = 0, conf = 0.999, 
                                                        use_sprt = ransac_use_sprt)
            
            output_dict['estimated_transform'] = est_transform.T'''
            #pred_len.append(output_dict['sg_matches_sparse'].shape[0])
            result_dict = evaluator(output_dict, batched_data)
            result_dict = release_cuda(result_dict)

            print("\rEvaluation on going ... {:3.2f}%".format((idx+1)/eval_num*100), end='')
            #print(result_dict)
            for k in result_dict_avgs:
                result_dict_avgs[k].append(result_dict[k])
            
    result_dict_logs =  {k: np.asarray(result_dict_avgs[k], dtype=np.float64) for k in result_dict_avgs}
    #print("how many prediction are made: ", sum(pred_len)/len(pred_len))
    model.train()
    # Get the mean of evaluation results
    for k in result_dict_avgs:
        result_dict_avgs[k] = np.asarray(result_dict_avgs[k]).mean()
    
    return result_dict_avgs,result_dict_logs
        

if __name__ == '__main__':
    from data.datasets import Scan3RDataset
    parse_args()
    from utils import common, scan3r
    import os.path as osp
    # Initialize Datasets
    dataset, val_dataset = get_datasets(args, cfg)
    #dataset = Scan3RDataset(dataset_root=cfg.dataset.root, split='val', use_augmentation=args.rand_trans)
    neighbor_limits =cfg.dataset.neighbor_limits

    for idx, output_dict in enumerate(dataset):
        #if idx > 100: break
        sg_pair = output_dict['sg']
        overlap = output_dict['overlap']
        subscans_scenes_dir = osp.join('/home/xie/Documents/datasets/3RScan/','out','scenes')
        subscans_files_dir = osp.join('/home/xie/Documents/datasets/3RScan/','out','files')
        mode = 'orig'
        nq, nt = sg_pair.x_q.shape[0], sg_pair.x_t.shape[0]
        if overlap <= 0.3 and nq < 10 and nt < 10 and nq >=6 and nt >= 6:
            if idx < 300: continue
            src_scan_id = output_dict['src_frame']
            ref_scan_id = output_dict['ref_frame']

            trans_gt = output_dict['transform']
            #src_points = remove_ceiling(output_dict['src_points'], 0.2)
            #ref_points = remove_ceiling(output_dict['ref_points'], 0.2)
            sg_match_gt = output_dict['sg_match']
            
            print()
            print("overlap", overlap)
            print(sg_match_gt.T)
            print('src', src_scan_id, 'ref', ref_scan_id)
            #print(sg_match_gt[:,0])
            #print(sg_match_gt[:,1])
            src_insts = output_dict['src_insts']
            ref_insts = output_dict['ref_insts']

            src_points, src_plydata = scan3r.load_plydata_npy(
                osp.join(subscans_scenes_dir, '{}/data.npy'.format(src_scan_id)), 
                obj_ids = None, return_ply_data=True)
            ref_points, ref_plydata = scan3r.load_plydata_npy(
                osp.join(subscans_scenes_dir, '{}/data.npy'.format(ref_scan_id)), 
                obj_ids = None, return_ply_data=True)
            
            src_data_dict = common.load_pkl_data(osp.join(subscans_files_dir, '{}/data/{}.pkl'.format(mode, src_scan_id)))
            ref_data_dict = common.load_pkl_data(osp.join(subscans_files_dir, '{}/data/{}.pkl'.format(mode, ref_scan_id)))

            src_color = np.vstack([src_plydata['red'], src_plydata['green'],src_plydata['blue']]).T.astype(np.float32)/255.
            ref_color = np.vstack([ref_plydata['red'], ref_plydata['green'],ref_plydata['blue']]).T.astype(np.float32)/255.
            
            pcd1 = make_open3d_point_cloud(src_points, src_color)
            pcd2 = make_open3d_point_cloud(ref_points, ref_color)
            #draw_registration_result(pcd1, pcd2, trans_gt, False)

            src_object_id2idx = {src_data_dict['object_id2idx'][key]: key for key in src_data_dict['object_id2idx']}
            ref_object_id2idx = {ref_data_dict['object_id2idx'][key]: key for key in ref_data_dict['object_id2idx']}
            
            edge_index_t = sg_pair.edge_index_t.cpu().numpy()
            edge_index_t_ids = []
            for ele in edge_index_t.T:
                edge_index_t_ids.append(np.asarray([src_object_id2idx[ele[0]], src_object_id2idx[ele[1]]]))
            edge_index_t_ids = np.asarray(edge_index_t_ids).T
            
            edge_index_q = sg_pair.edge_index_q.cpu().numpy()
            edge_index_q_ids = []
            for ele in edge_index_q.T:
                edge_index_q_ids.append(np.asarray([ref_object_id2idx[ele[0]], ref_object_id2idx[ele[1]]]))
            edge_index_q_ids = np.asarray(edge_index_q_ids).T
            
            nyu_ids = np.unique(src_plydata['NYU40'])
            print(nyu_ids)
            print([NYU40_Label_Names[nyu_id] for nyu_id in nyu_ids])
            nyu_ids = np.unique(ref_plydata['NYU40'])
            print(nyu_ids)
            print([NYU40_Label_Names[nyu_id] for nyu_id in nyu_ids])
            
            draw_scene_graph_with_pcd(src_points, src_color, 
                                      src_plydata['objectId'], edge_index_t_ids)
            draw_scene_graph_with_pcd(src_points, src_color, 
                                      src_plydata['objectId'], edge_index_t_ids, 
                                      draw_pcd=False)
            draw_point_cloud(pcd1)

            draw_scene_graph_with_pcd(ref_points, ref_color, 
                                      ref_plydata['objectId'], edge_index_q_ids)
            draw_scene_graph_with_pcd(ref_points, ref_color, 
                                      ref_plydata['objectId'], edge_index_q_ids, 
                                      draw_pcd=False)
            draw_point_cloud(pcd2)

            

            
            
            #print(src_object_id2idx)
            #print(ref_object_id2idx)

            src_color_new = np.ones_like(src_color)*0.5
            for ele in sg_match_gt[:,1]:
                ele_id = src_object_id2idx[ele.item()]
                src_color_new[src_plydata['objectId'] == ele_id, :] = np.asarray([1, 0.706, 0])
            ref_color_new = np.ones_like(ref_color)*0.5
            for ele in sg_match_gt[:,0]:
                ele_id = ref_object_id2idx[ele.item()]
                ref_color_new[ref_plydata['objectId'] == ele_id, :] = np.asarray([0, 0.651, 0.929])

            pcd1.colors = o3d.utility.Vector3dVector(src_color_new)
            pcd2.colors = o3d.utility.Vector3dVector(ref_color_new)
            draw_registration_result(pcd1, pcd2, trans_gt, False)
            draw_registration_result(pcd1, pcd2, trans_gt, True)



            

        print('\r', idx, end='')

    print("Dataset loaded.")

