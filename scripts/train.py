import os
import datetime, time
import argparse
import torch
import torch.utils.data

from net import SGPGM_GeoTransformer
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

loss_types = ['loss','s_l', 'k_l'] 
eval_types = ['Overlap', 'IR', 'CCD', 'RRE', 'RTE', 'FMR', 'RMSE', 'RR', 'CorrS',
              'HIT1', 'HIT3', 'HIT5', 'MRR', 'F1']

def parse_args(argv=None):
    parser  = argparse.ArgumentParser(description='3D Scene Graph Macthing Training')

    # Hyper Parameters for Training
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training.')
    parser.add_argument('--warmup_until', type=int, default=5000, help='Warm up by linearly interpolating the learning rate until some iterations.')
    parser.add_argument('--lr_warmup', type=float, default=1e-6, help='Warm up by linearly interpolating the learning rate from some smaller value.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs for training.')
    parser.add_argument('--decay_epochs', type=int, default=4, help='Decrease the learning rate by 0.1 every N epochs.')

    # Basic Settings
    parser.add_argument('--config', type=str, default="./data/sg_pgm.json", help='Configuration to be loaded.')
    parser.add_argument('--dataset_root', type=str, default="/home/xie/Documents/datasets/3RScan/dp_fast", help='Path for load dataset.')
    parser.add_argument('--dataset', choices=['sgm', 'sgm_tar'], default='sgm_tar', help='Path for load dataset.')
    parser.add_argument('--num_workers', default=12, type=int, help='')
    parser.add_argument('--split',  choices=['train', 'valid', 'all'], default='all', help='' )
    parser.add_argument('--device',  choices=['cuda', 'cpu'], default='cuda', help='Device to train on.' )
    parser.add_argument('--reproducibility', default=True, action='store_true', help='Set true if you want to reproduct the almost same results as given in the ablation study.') 

    # Training Settings
    parser.add_argument('--validation_epoch', default=1, type=int, help='Output validation every n iterations. If -1, do no validation.')
    parser.add_argument('--eval_num', default=-1, type=int, help='Sample numbers to use for evaluation.')
    parser.add_argument('--save_folder', type=str, default="./weights", help='Path for saving training weights.')
    parser.add_argument('--log_root', type=str, default='./logs', help='Path for saving training logs.')
    parser.add_argument('--pretrained_3dmatch', type=str, default="./weights/geotransformer-3dmatch.pth.tar", help='Path for pretrained weights on 3dmatch.')
    parser.add_argument('--interrupt', type=bool, default=True, help='Save weights when interreupted.')
    parser.add_argument('--rand_trans', default=False, action='store_true', help='augmented random transformation between src and ref fragments.')
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
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    global cur_lr
    cur_lr = new_lr

def train(cfg):
    # Initialize Network, Loss and Optimizer
    model = SGPGM_GeoTransformer(cfg).cuda()
    model.load_pretrained(args.pretrained_3dmatch)
    model.lock_geotr()
    criterion = OverallLoss(cfg=cfg)
    evaluator = Evaluator(cfg=cfg, eval_coarse=False, eval_geo=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Model loaded.")

    # Initialize TensorBoardX Writer
    model_name = model.model_name
    begin_time = (datetime.datetime.now()).strftime("%d%m%Y%H%M%S")
    variant_name = args.save_folder.split('/')[-1] if args.save_folder != "./weights" else "unnamed"
    logpath = os.path.join(args.log_root, (model_name + "_" + variant_name + "_" + begin_time))
    logger = SGMatchLogger(logpath, args)
    logger.log_model(model)
    
    
    # Initialize Datasets
    train_dataset, val_dataset = get_datasets(args, cfg)
    train_loader, val_loader = get_dataloader(train_dataset, val_dataset, cfg, args)
    print("Dataset loaded.")

    epoch_size = train_dataset.__len__()//args.batch_size
    max_iter = epoch_size * args.epochs

    # Initialize accumulator and timer
    time_avg = MovingAverage()
    loss_avgs = {k: MovingAverage(100) for k in (loss_types)}

    last_time = time.time()
    # Start Training
    iteration = 0

    try:
        print("Start training.")
        for epoch in range(args.epochs):
            # Decay learning rate after N epochs by 0.1
            if epoch % args.decay_epochs == 0 and epoch >1:
                set_lr(optimizer, args.lr* 0.1**(epoch//args.decay_epochs))

            for idx, batched_data in enumerate(train_loader):

                # Warm up by linearly interpolating the learning rate from some smaller value
                if args.warmup_until > 0 and iteration <= args.warmup_until:
                    set_lr(optimizer, (args.lr - args.lr_warmup) * (iteration / args.warmup_until) + args.lr_warmup)
                
                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Network forward
                batched_data = to_cuda(batched_data)
                output_dict = model(batched_data)
                
                loss_dict = criterion(output_dict, batched_data)
                batched_data = torch_utils.release_cuda(batched_data)
                output_dict = torch_utils.release_cuda(output_dict)
                logger.log_train_loss(loss_dict, iteration)
                
                if torch.isfinite(loss_dict['loss']).item():
                    loss_dict['loss'].backward()
                    optimizer.step()
                    loss_dict = torch_utils.release_cuda(loss_dict)
                    for k in loss_avgs:
                        if np.isfinite(loss_dict[k]):
                            loss_avgs[k].add(loss_dict[k])
                else: 
                    print("meet infinite", loss_dict)
                
                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != 0:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = str(datetime.timedelta(seconds=(max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                    #loss_labels = sum([[(k +  ":"),round(loss_avgs[k].get_avg(), 3)] for k in (loss_types+eval_types) if k in (loss_types+eval_types)], [])
                    loss_labels = sum([[(k +  ":"),round(loss_avgs[k].get_avg(), 3)] for k in (loss_types) if k in (loss_types)], [])
                    print(('\r[%3d] %7d ||' + (' %s %.3f |' * len(loss_avgs)) + '| ETA: %s')
                                % tuple([epoch, iteration] + loss_labels + [eta_str]), flush=True, end='')

                iteration = iteration + 1

            # Save weights after every epoch
            model.save_weights(args.save_folder, epoch, iteration)
            # Evaluation
            if epoch < args.epochs - 1 and (epoch+1) % args.validation_epoch == 0:
                print()
                eval_results = evaluation(model=model, val_loader=val_loader, evaluator=evaluator, logger=logger, epoch=epoch)
                logger.log_eval_metrics(eval_results, iteration)
                print()
                print("Evaluation Metrics for epoch {}".format(epoch))
                print('   '.join('{}: {:5.4f}'.format(k, eval_results[k]) for k in eval_results))

        print()
        eval_results = evaluation(model=model, val_loader=val_loader, evaluator=evaluator, logger=logger, epoch=epoch, eval_num=args.eval_num)
        logger.log_eval_metrics(eval_results, iteration)
        print()
        print("Evaluation Metrics for epoch {}".format(epoch))
        print('   '.join('{}: {:5.4f}'.format(k, eval_results[k]) for k in eval_results))
        model.save_weights(args.save_folder, epoch, iteration)

    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            model.save_weights(args.save_folder, epoch, iteration)

def evaluation(model, val_loader, evaluator: Evaluator, logger: SGMatchLogger, epoch, eval_num=-1, vis_num=8):
    model.eval()
    if eval_num < 0: eval_num = val_loader.__len__()
    result_dict_avgs = {k: [] for k in eval_types}
    with torch.no_grad():
        for idx, batched_data in enumerate(val_loader):
            batched_data = to_cuda(batched_data)
            output_dict = model(batched_data, early_stop=True)
            batched_data = torch_utils.release_cuda(batched_data)
            output_dict = torch_utils.release_cuda(output_dict)
            result_dict = evaluator(output_dict, batched_data)
            result_dict = release_cuda(result_dict)
            print("\rEvaluation on going ... {:3.2f}%".format((idx+1)/eval_num*100), end='')
            for k in result_dict_avgs:
                result_dict_avgs[k].append(result_dict[k])
            if idx >= eval_num:
                break
    model.train()
    # Get the mean of evaluation results
    for k in result_dict_avgs:
        result_dict_avgs[k] = np.asarray(result_dict_avgs[k]).mean()
    
    return result_dict_avgs
        

if __name__ == '__main__':
    parse_args()
    train(cfg=cfg)

