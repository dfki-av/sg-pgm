import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GATConv, GATv2Conv


class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.A = nn.parameter.Parameter(torch.Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.shape[-1] == Y.shape[-1] == self.d
        M = torch.matmul(X, (self.A + self.A.transpose(0, 1).contiguous())/2)
        M = torch.matmul(M, Y.transpose(1, 2).contiguous())
        return M

class Affinity_Double(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d1, d2):
        super(Affinity_Double, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.A = nn.parameter.Parameter(torch.Tensor(self.d1, self.d1))
        self.B = nn.parameter.Parameter(torch.Tensor(self.d2, self.d2))
        self.reset_parameters()
        # TODO:

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d1)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d1)
        stdv = 1. / math.sqrt(self.d2)
        self.B.data.uniform_(-stdv, stdv)
        self.B.data += torch.eye(self.d2)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.shape[-1] == Y.shape[-1] == (self.d1 + self.d2)
        M_1 = torch.matmul(X[:,:,:self.d1], (self.A + self.A.transpose(0, 1).contiguous())/2)
        M_2 = torch.matmul(X[:,:,self.d1:], (self.B + self.B.transpose(0, 1).contiguous())/2)
        M = torch.matmul(M_1, Y[:,:,:self.d1].transpose(1, 2).contiguous()) + torch.matmul(M_2, Y[:,:,self.d1:].transpose(1, 2).contiguous())
        
        return M