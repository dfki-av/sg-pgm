import torch
import torch.utils.data
if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    sys.path.append('../../')
from sg_pgm import SGPGM_GeoTransformer
from data.stack_mode import registration_collate_fn_stack_mode, build_dataloader_stack_mode
import data.torch_utils as torch_utils 

from utils.pointcloud import *
from data.torch_utils import to_cuda, release_cuda
from modules.loss.loss import Evaluator
from utils import common
from utils.net_args import parse_args

EVAL_TYPES = ['Overlap', 'IR', 'CCD', 'RRE', 'RTE', 'FMR', 'RMSE', 'RR', 'CorrS',
              'HIT1', 'HIT3', 'HIT5', 'MRR', 'F1']

def evaluation(model, val_loader, evaluator: Evaluator, eval_num=-1, eval_types=EVAL_TYPES):
    model.eval()
    
    if eval_num < 0: eval_num = val_loader.__len__()
    result_dict_avgs = {k: [] for k in eval_types}
    
    with torch.no_grad():
        for idx, batched_data in enumerate(val_loader):
            if idx >= eval_num: break
            
            if batched_data['sg_match'][0].shape[0] == 0:
                for k in result_dict_avgs:
                    result_dict_avgs[k].append(0)
                continue
            
            batched_data = to_cuda(batched_data)
            output_dict = model(batched_data, early_stop=False)

            batched_data = torch_utils.release_cuda(batched_data)
            output_dict = torch_utils.release_cuda(output_dict)

            result_dict = evaluator(output_dict, batched_data)
            result_dict = release_cuda(result_dict)

            print("\rEvaluation on going ... {:3.2f}%".format((idx+1)/eval_num*100), end='')
            for k in result_dict_avgs:
                result_dict_avgs[k].append(result_dict[k])
            
    result_dict_logs =  {k: np.asarray(result_dict_avgs[k], dtype=np.float64) for k in result_dict_avgs}
    # Get the mean of evaluation results
    for k in result_dict_avgs:
        result_dict_avgs[k] = np.asarray(result_dict_avgs[k]).mean()
    
    return result_dict_avgs,result_dict_logs
        

if __name__ == '__main__':
    from data.datasets import Scan3RDataset
    args, cfg = parse_args()

    # Initialize Network, Loss and Optimizer
    model = SGPGM_GeoTransformer(cfg).cuda()
    model.load_weights(args.trained_model, args.pretrained_3dmatch)
    print("Model loaded from: ", args.trained_model, args.pretrained_3dmatch)
    evaluator = Evaluator(cfg=cfg, eval_coarse=False, eval_geo=True)

    # Initialize Datasets
    val_set = Scan3RDataset(dataset_root=cfg.dataset.root, split='val', 
                            use_augmentation=args.rand_trans, predicted=False)

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

    eval_results, result_dict_logs = evaluation(model=model, val_loader=val_loader, evaluator=evaluator, eval_num=args.eval_num)

    print()
    print("Evaluation Results:")
    bins = [0.1, 0.3, 0.6]
    inds = np.digitize(result_dict_logs['Overlap'], bins)
    bins = bins + [1.0]
    for i in range(len(bins)-1):
        inlier_inds = inds==(i+1)
        result_dict_logs['Overlap'][inlier_inds]
        bined_results = {'Overlap': str(round(bins[i], 2)) + '-' + str(round(bins[i+1], 2))}
        for k in result_dict_logs:
            if k != "Overlap":
                bined_results[k] = round(result_dict_logs[k][inlier_inds].mean(), 4)
        print('   '.join('{}: {}'.format(k, bined_results[k]) for k in bined_results))

    print('   '.join('{}: {:5.4f}'.format(k, eval_results[k]) for k in eval_results))

