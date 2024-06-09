import os
import json
import torch
from tensorboardX import SummaryWriter

class SGMatchLogger():
    def __init__(self, logpath, args) -> None:
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        self.root = logpath
        self.writer = SummaryWriter(logpath)
        self.args = args

    def log_model(self, model, cfg=None):
        model_dict = {}
        train_dict = {}
        model_dict['model'] = {}
        for name, param in model.state_dict().items():
            model_dict['model'][name] = param.shape
        cnt = 0
        for param in model.parameters():
            cnt=cnt+torch.numel(param)
        model_dict["model_size"] = cnt

        train_dict["batch_size"] = self.args.batch_size
        train_dict["lr"] = self.args.lr

        with open(os.path.join(self.root, "train_cfg.json"), 'w') as file:
            json.dump(train_dict, file)
        with open(os.path.join(self.root, "model.json"), 'w') as file:
            json.dump(model_dict, file)
    
    def log_train_loss(self, losses, iteration):
        for k in losses:
            self.writer.add_scalar('Loss/train/'+k, losses[k], iteration)

    def log_eval_metrics(self, metrics, iteration):
        for k in metrics:
            self.writer.add_scalar("Eval/valid/"+k, metrics[k], iteration)