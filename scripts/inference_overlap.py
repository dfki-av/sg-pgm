import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append('.')
import torch
import torch.utils.data
import json

from sklearn.metrics import confusion_matrix
from sg_pgm import SGPGM_GeoTransformer
from data.stack_mode import registration_collate_fn_stack_mode, build_dataloader_stack_mode
import data.torch_utils as torch_utils
from utils.pointcloud import *
from utils.net_args import parse_args

def compute_precision_recall(result_dict):
        tn, fp, fn, tp = confusion_matrix(result_dict['true'], result_dict['pred'], labels=[0, 1]).ravel()
        precision = round(tp / (tp + fp), 4)
        recall = round(tp / (tp + fn), 4)
        f1_score = round(2 * (precision * recall)/ (precision + recall), 4)
        metrics_dict = {'precision' : precision, 'recall' : recall, 'f1_score' : f1_score}

        return metrics_dict

def evaluation(model, val_loader, args, cfg, eval_num=-1):
    model.eval()
    
    if eval_num < 0: eval_num = val_loader.__len__()
    overlapper_data = {'true' : [], 'pred' : []}
    records = {'overlap': [], 'global_score':[], 'node_scores': [], 'overlap_score': []}
    with torch.no_grad():
        for idx, batched_data in enumerate(val_loader):
            if idx >= eval_num: break

            overlap = batched_data['overlap']

            batched_data = torch_utils.to_cuda(batched_data)
            output_dict = model(batched_data, early_stop=True)
            sg_matches = output_dict['sg_matches_raw']
            sg_matches_sparse = output_dict['sg_matches_sparse']
            sg_matches_sparse_scores = sg_matches[:,sg_matches_sparse[:,0], sg_matches_sparse[:,1]]

            batched_data = torch_utils.release_cuda(batched_data)
            output_dict = torch_utils.release_cuda(output_dict)
            
            k = output_dict['top_k']
            if args.at3:
                topk = min(3, int(sg_matches_sparse.shape[0]))
                c_topk, _ = torch.topk(sg_matches_sparse_scores.squeeze(), topk)
                overlap_score = k * c_topk.mean()
            else:
                overlap_score = sg_matches_sparse_scores.mean() 
            
            records['node_scores'].append(sg_matches_sparse_scores.cpu().tolist())
            records['overlap'].append(overlap)
            records['global_score'].append(k)
            records['overlap_score'].append(overlap_score.cpu().item())
            overlapper_data['true'].append(1.0 if overlap > 0.0 else 0.0)
            if args.at3:
                overlapper_data['pred'].append(1.0 if overlap_score > cfg.eval.overlap_threshold_at3 else 0.0)
            else:
                overlapper_data['pred'].append(1.0 if overlap_score > cfg.eval.overlap_threshold else 0.0)

            print("\rEvaluation on going ... {:3.2f}%".format((idx+1)/eval_num*100), end='')

        with open("overlap_results_k.json", 'w') as file: json.dump(records, file)
    return overlapper_data

if __name__ == '__main__':
    from data.datasets import Scan3RDataset
    args, cfg = parse_args()
    
    # Initialize Network, Loss and Optimizer
    model = SGPGM_GeoTransformer(cfg).cuda()
    model.load_weights(args.trained_model, args.pretrained_3dmatch)
    print("Model loaded from: ", args.trained_model, args.pretrained_3dmatch)

    # Initialize Datasets
    val_set = Scan3RDataset(dataset_root=cfg.dataset.root, split='val', 
                            use_augmentation=args.rand_trans, 
                            anchor_type_name="_subscan_anchors_w_wo_overlap")
    print(val_set.__len__())

    val_loader = build_dataloader_stack_mode(
        val_set,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        cfg.dataset.neighbor_limits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        distributed=False,
        reproducibility=args.reproducibility,
        point_limits=cfg.dataset.max_c_points
    )

    print("Dataset loaded.")

    eval_results = evaluation(model=model, val_loader=val_loader, args=args, cfg=cfg, eval_num=args.eval_num)
    metrics_dict = compute_precision_recall(eval_results)

    print("Evaluation Results:")
    print('   '.join('{}: {:5.4f}'.format(k, metrics_dict[k]) for k in metrics_dict))