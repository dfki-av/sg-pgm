import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GATConv, GATv2Conv

class GATEncoder(nn.Module):
    def __init__(self, hidden_dim: int, recep_field: int, node_input_dim:int, edge_input_dim:int, gnn_type: str="GATv2", skip_flag: bool=True,  feat_mlp: bool = True):
        super().__init__()
        self.use_feat_mlp= feat_mlp
        self.node_feat_enc_dims = [hidden_dim]
        self.edge_feat_enc_dims = [hidden_dim]
        self.recep_field = recep_field
        
        if self.use_feat_mlp:
            self.node_feat_mlp = self._build_feat_mlp(input_dim=node_input_dim, feat_dims = self.node_feat_enc_dims)
            self.edge_feat_mlp = self._build_feat_mlp(input_dim=edge_input_dim, feat_dims = self.edge_feat_enc_dims)

        self.skip_connections = skip_flag

        if self.skip_connections:
            self.learnable_skip = nn.Parameter(torch.ones(self.recep_field, self.recep_field))

        self.conv_layers = nn.ModuleList()
        conv_model = self._build_conv_model(model_type=gnn_type)
        for i in range(self.recep_field):
            if self.skip_connections:
                hidden_input_dim = hidden_dim * (i + 1)
            else:
                hidden_input_dim = hidden_dim
            self.conv_layers.append(conv_model(in_channels=hidden_input_dim, out_channels=hidden_dim, edge_dim=hidden_dim))

    def _build_feat_mlp(self, input_dim: int, feat_dims: list):
        layer = []
        layer.append(nn.Linear(input_dim, feat_dims[0]))
        for i in range(1, len(feat_dims)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(feat_dims[i - 1], feat_dims[i]))
        layer.append(nn.ReLU())
        return nn.Sequential(*layer)
    
    def _build_conv_model(self, model_type):
        if model_type == "GAT":
            return GATConv
        elif model_type == "GATv2":
            return GATv2Conv
        else:
            print("unrecognized model type")
    
    def forward(self, node_feat, edge_index, edge_feat):
    
        # MLP for node and edge features
        if self.use_feat_mlp:
            node_emb = self.node_feat_mlp(node_feat)
            edge_emb = self.edge_feat_mlp(edge_feat)
        else:
            node_emb = node_feat
            edge_emb = edge_feat
        
        if self.skip_connections:
            all_emb = node_emb.unsqueeze(dim=1)
            node_num = node_emb.shape[0]
        
        # Stack conv layers
        for i in range(self.recep_field):
            if self.skip_connections:
                skip_vals = self.learnable_skip[i,:i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(node_num, -1)
                
            node_emb = F.relu(self.conv_layers[i](node_emb, edge_index, edge_emb))
            
            if self.skip_connections:
                all_emb = torch.cat((all_emb, node_emb.unsqueeze(1)), 1)
                node_emb = torch.cat((node_emb, curr_emb), dim=1)

        return node_emb