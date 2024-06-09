import torch
import torch.nn.functional as F
import torch_geometric as pyg
import numpy as np
import random
import open3d as o3d

def set_reproducibility(seed=1000, report=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    pyg.seed.seed_everything(seed)

    if report:
        print()
        print('************************Reproducibility Mode******************************')
        print('**                                                                      **')
        print('** Set the random seed for random, np.random, torch and cudnn as {:4d}.  **'.format(seed))
        print('**                                                                      **')
        print('**************************************************************************')
        print()
