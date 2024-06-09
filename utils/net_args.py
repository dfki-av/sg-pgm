import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append('.')

import argparse
from data.config import Config
from utils.set_seed import set_reproducibility

def parse_args(argv=None):
    parser  = argparse.ArgumentParser(description='3D Scene Graph Macthing Testing')

    # Hyper Parameters for Training
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training.')
    parser.add_argument('--warmup_until', type=int, default=5000, help='Warm up by linearly interpolating the learning rate until some iterations.')
    parser.add_argument('--lr_warmup', type=float, default=1e-6, help='Warm up by linearly interpolating the learning rate from some smaller value.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs for training.')
    parser.add_argument('--decay_epochs', type=int, default=4, help='Decrease the learning rate by 0.1 every N epochs.')
    
    # Basic Settings
    parser.add_argument('--config', type=str, default="./data/sg_pgm.json", help='Configuration to be loaded.')
    parser.add_argument('--dataset_root', type=str, default="/home/xie/Documents/datasets/3RScan", help='Path for load dataset.')
    parser.add_argument('--dataset', choices=['sgm', 'sgm_tar', 'sgm_pointnet'], default='sgm', help='Path for load dataset.')
    parser.add_argument('--num_workers', default=12, type=int, help='')
    parser.add_argument('--device',  choices=['cuda', 'cpu'], default='cuda', help='Device to train on.' )
    parser.add_argument('--reproducibility', default=True, action='store_true', help='Set true if you want to reproduct the almost same results as given in the ablation study.') 

    # Training Settings
    parser.add_argument('--validation_epoch', default=1, type=int, help='Output validation every n iterations. If -1, do no validation.')
    parser.add_argument('--eval_num', default=-1, type=int, help='Sample numbers to use for evaluation.')
    parser.add_argument('--rand_trans', default=False, action='store_true', help='augmented random transformation between src and ref fragments.')
    parser.add_argument('--save_folder', type=str, default="./weights", help='Path for saving training weights.')
    parser.add_argument('--log_root', type=str, default='./logs', help='Path for saving training logs.')
    parser.add_argument('--interrupt', type=bool, default=True, help='Save weights when interreupted.')
    parser.add_argument('--pretrained_3dmatch', type=str, default="./weights/geotransformer-3dmatch.pth.tar", help='Path for pretrained weights on 3dmatch.')

    # Inference Settings
    parser.add_argument('--trained_model', type=str, default=None, help='Path of the trained weights.')
    
    # Overlap Evaluation
    parser.add_argument('--at3', default=True, action='store_true', help='Whether to use overlap_score_at3 evaluation.')
    

    args = parser.parse_args(argv)

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
    
    
    return args, cfg

