import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append('.')

import datetime, time
import argparse
import torch
import torch.utils.data

from sklearn.metrics import confusion_matrix

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

EVAL_TYPES = ['Overlap', 'IR', 'CCD', 'RRE', 'RTE', 'FMR', 'RMSE', 'RR', 'CorrS',
              'HIT1', 'HIT3', 'HIT5', 'MRR', 'F1']

def parse_args(argv=None):
    parser  = argparse.ArgumentParser(description='3D Scene Graph Macthing Testing')
    # Hyper Parameters for Training
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')

    # Basic Settings
    parser.add_argument('--config', type=str, default="./data/sg_pgm.json", help='Configuration to be loaded.')
    parser.add_argument('--dataset_root', type=str, default="/home/xie/Documents/datasets/3RScan", help='Path for load dataset.')
    parser.add_argument('--dataset', choices=['sgm', 'sgm_tar', 'sgm_pointnet'], default='sgm', help='Path for load dataset.')
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
        set_reproducibility()
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

def compute_precision_recall(result_dict):
        tn, fp, fn, tp = confusion_matrix(result_dict['true'], result_dict['pred'], labels=[0, 1]).ravel()
        precision = round(tp / (tp + fp), 4)
        recall = round(tp / (tp + fn), 4)
        f1_score = round(2 * (precision * recall)/ (precision + recall), 4)
        metrics_dict = {'precision' : precision, 'recall' : recall, 'f1_score' : f1_score}

        return metrics_dict

def evaluation(model, val_loader, evaluator: Evaluator, 
               logger: SGMatchLogger=None, epoch=None, eval_num=-1, eval_types=EVAL_TYPES):
    model.eval()
    
    if eval_num < 0: eval_num = val_loader.__len__()
    result_dict_avgs = {k: [] for k in eval_types}

    overlapper_data = {'true' : [], 'pred' : []}

    with torch.no_grad():
        for idx, batched_data in enumerate(val_loader):
            if idx >= eval_num: break

            overlap = batched_data['overlap']
            batched_data = to_cuda(batched_data)
            output_dict = model(batched_data, early_stop=True)
            sg_pair = batched_data['sg']
            #print(sg_pair.x_q.shape, sg_pair.edge_index_q.shape, sg_pair.edge_attr_q.shape)
            #print(sg_pair.x_q.sum(dim=-1))

            batched_data = torch_utils.release_cuda(batched_data)
            output_dict = torch_utils.release_cuda(output_dict)
            
            k = output_dict['top_k']
            sg_matches = output_dict['sg_matches_raw']
            sg_matches_sparse = output_dict['sg_matches_sparse']

            overlap_score = k * sg_matches[:,sg_matches_sparse[:,0], sg_matches_sparse[:,1]].sum()/ sg_matches_sparse.shape[0]
            
            overlapper_data['pred'].append(1.0 if overlap_score > 0.375 else 0.0)
            overlapper_data['true'].append(1.0 if overlap > 0.0 else 0.0)
            #result_dict = evaluator(output_dict, batched_data)
            #result_dict = release_cuda(result_dict)
            
            print("\rEvaluation on going ... {:3.2f}%".format((idx+1)/eval_num*100), end='')
            #print(result_dict)
            #for k in result_dict_avgs:
                #result_dict_avgs[k].append(result_dict[k])
    return overlapper_data
    result_dict_logs =  {k: np.asarray(result_dict_avgs[k], dtype=np.float64) for k in result_dict_avgs}
    
    model.train()
    # Get the mean of evaluation results
    for k in result_dict_avgs:
        result_dict_avgs[k] = np.asarray(result_dict_avgs[k]).mean()
    
    return result_dict_avgs,result_dict_logs
        



if __name__ == '__main__':
    from data.datasets import Scan3RDataset
    parse_args()
    
    # Initialize Network, Loss and Optimizer
    model = GeoTransformer(cfg, args.ptfusion, args.sgfusion).cuda()
    model.load_weights(args.trained_model, args.pretrained_3dmatch)
    evaluator = Evaluator(cfg=cfg, eval_coarse=False)

    # Initialize Datasets
    val_set = Scan3RDataset(dataset_root=cfg.dataset.root, split='val', 
                            use_augmentation=args.rand_trans, 
                            anchor_type_name="_subscan_anchors_w_wo_overlap")
    print(val_set.__len__())
    neighbor_limits =cfg.dataset.neighbor_limits

    val_loader = build_dataloader_stack_mode(
        val_set,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        distributed=False,
        reproducibility=args.reproducibility,
        point_limits=args.max_c_points
    )

    print("Dataset loaded.")
    
    eval_results = evaluation(model=model, val_loader=val_loader, evaluator=evaluator, eval_num=args.eval_num)
    metrics_dict = compute_precision_recall(eval_results)

    print("Evaluation Results:")
    print('   '.join('{}: {:5.4f}'.format(k, metrics_dict[k]) for k in metrics_dict))

