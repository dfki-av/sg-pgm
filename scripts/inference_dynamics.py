import torch
import torch.utils.data

from sg_pgm import SGPGM_GeoTransformer
from data.stack_mode import registration_collate_fn_stack_mode, build_dataloader_stack_mode
import data.torch_utils as torch_utils 

from utils.pointcloud import *
from modules.loss.loss import  Evaluator
from utils.net_args import parse_args

EVAL_TYPES = ['Overlap', 'HIT1', 'HIT3', 'HIT5', 'MRR', 'F1']

DYNAMICS = ['sub2rescan', 'sub2scan', 'sub2sub']
ANCHOR_TYPE_NAME = ["subscan_to_refscan_changed", "subscan_to_scan", "subscan_to_subscan_changed"]

def evaluation(model, val_loader, evaluator: Evaluator, eval_types=EVAL_TYPES):
    
    model.eval()
    eval_num = val_loader.__len__()
    result_dict_avgs = {k: [] for k in eval_types}

    with torch.no_grad():
        for idx, batched_data in enumerate(val_loader):
            if idx >= eval_num: break

            batched_data = torch_utils.to_cuda(batched_data)
            output_dict = model(batched_data, early_stop=True)

            batched_data = torch_utils.release_cuda(batched_data)
            output_dict = torch_utils.release_cuda(output_dict)
            
            result_dict = evaluator(output_dict, batched_data)
            result_dict = torch_utils.release_cuda(result_dict)

            print("\rEvaluation on going ... {:3.2f}%".format((idx+1)/eval_num*100), end='')
            
            for k in result_dict_avgs:
                result_dict_avgs[k].append(result_dict[k])
            
    result_dict_logs =  {k: np.asarray(result_dict_avgs[k], dtype=np.float64) for k in result_dict_avgs}

    # Get the mean of evaluation results
    for k in result_dict_avgs:
        result_dict_avgs[k] = np.asarray(result_dict_avgs[k]).mean()
    
    return result_dict_avgs,result_dict_logs
        

if __name__ == '__main__':
    from data.datasets import Scan3RDataset_Dynamics
    args, cfg = parse_args()

    anchor_type_name = ANCHOR_TYPE_NAME[DYNAMICS.index(args.dynamics)]

    # Initialize Network, Loss and Optimizer
    model = SGPGM_GeoTransformer(cfg).cuda()
    model.load_weights(args.trained_model, args.pretrained_3dmatch)
    evaluator = Evaluator(cfg=cfg, eval_coarse=False, eval_geo=False)

    # Initialize Datasets
    val_set = Scan3RDataset_Dynamics(dataset_root=cfg.dataset.root, 
                                     split='val', 
                                     anchor_type_name=anchor_type_name, #'subscan_to_refscan_changed',
                                     use_augmentation=args.rand_trans)

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

    print("Dataset loaded, total samples {}".format(val_set.__len__()))

    eval_results, result_dict_logs = evaluation(model=model, val_loader=val_loader, 
                                                evaluator=evaluator)

    print("Evaluation Results:")
    print('   '.join('{}: {:5.4f}'.format(k, eval_results[k]) for k in eval_results))
    