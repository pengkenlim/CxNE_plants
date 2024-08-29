#setting sys.path for importing modules
import os
import sys

if __name__ == "__main__":
         abspath= __file__
         parent_module= "/".join(abspath.split("/")[:-2])
         sys.path.insert(0, parent_module)

"""Script for training and evaluating your GCN model
I will usee this script to test out GAT"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_cluster import random_walk
from torch_geometric.nn.models import GAT 
from torch_geometric.data import Data
import os
import sys
import numpy as np
import scipy.sparse as sp
from torch_geometric.loader import NeighborSampler ,GraphSAINTSampler , GraphSAINTRandomWalkSampler
import torch.nn.init as init

def adjacency_matrix_to_edge_index(adjacency_matrix):
    coo_matrix = sp.coo_matrix(adjacency_matrix)
    row_indices = torch.tensor(coo_matrix.row, dtype=torch.long)
    col_indices = torch.tensor(coo_matrix.col, dtype=torch.long)
    edge_index = torch.stack([row_indices, col_indices], dim=0)
    return edge_index

def init_all_layers():
    for i in range(0,num_layers):
        if hasattr(model.convs[i], 'lin_l') and model.convs[i].lin_l is not None:
            init.kaiming_uniform_(model.convs[i].lin_l.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(model.convs[i].lin_l, 'bias') and model.convs[i].lin_l.bias is not None:
                init.zeros_(model.convs[i].lin_l.bias)
        if hasattr(model.convs[i], 'lin_r') and model.convs[i].lin_r is not None:
            init.kaiming_uniform_(model.convs[i].lin_r.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(model.convs[i].lin_r, 'bias') and model.convs[i].lin_r.bias is not None:
               init.zeros_(model.convs[i].lin_r.bias)

#parameters
###########################
workdir_main = "/mnt/md2/ken/CxNE_plants_data/models"
workdir_root = "/mnt/md2/ken/CxNE_plants_data/"
model_name = "model_1"
num_workers = 60
start_epoch = 1
end_epoch = 31 #exclude 31
learning_rate = 1e-3
batch_size = 64
act = "relu"
neighbourhood_sizes = [500,500]
dropout_rate = 0.05
v2 = True
concat =True
heads = 2
hidden_channels = 256
num_layers = 2
out_channels = None
device = torch.device('cpu')
species = "taxid3702"
################################

if not os.path.exists(workdir_main):
    os.makedirs(workdir_main)

model = GAT(in_channels = 480, 
            hidden_channels = hidden_channels, 
            num_layers  = num_layers, 
            out_channels  = out_channels, 
            act =act, v2 = v2,
            concat =concat, 
            heads = heads)
init_all_layers()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
logpath = os.path.join(workdir_main, model_name +".txt")

########################################
#load nodes_split
species_node_split_idx_path = os.path.join(workdir_main, "species_node_split_idx.pkl")
with open(species_node_split_idx_path, "rb") as f:
    species_node_split_idx= pickle.load(f)
node_split_idx = species_node_split_idx[species]

#load genedict
gene_dict_path = os.path.join(workdir_root, species, "gene_dict.pkl")
with open(gene_dict_path, "rb") as f:
    gene_dict= pickle.load(f)

#load node features
node_features_path = os.path.join(workdir_root, species,  "node_features.pkl")
with open(node_features_path, "rb") as f:
    node_features= pickle.load(f)
node_features  = torch.tensor(node_features,dtype=torch.float )

#load adj
adj_mat_zscore_norm_ranks_bin1000_symmetrical_path = os.path.join(workdir_root, species,"adj_mat_zscore_norm_ranks_bin1000_symmetrical.pkl")
with open(adj_mat_zscore_norm_ranks_bin1000_symmetrical_path, "rb") as f:
    adj_mat_zscore_norm_ranks_bin1000_symmetrical= pickle.load(f)

#make data
data = Data(x=node_features, edge_index= adjacency_matrix_to_edge_index(adj_mat_zscore_norm_ranks_bin1000_symmetrical))

optimizer.zero_grad()
out = model.forward(data.x, data.edge_index).cpu() 





######################################
help(model.forward)

#train_loaders = NeighborSampler(data.edge_index, sizes= neighbourhood_sizes, batch_size= 1,
#                                shuffle=True, node_idx = node_split_idx["train_node_idx"] , num_nodes = data.num_nodes, num_workers = num_workers)

train_loader_graphsaint = GraphSAINTRandomWalkSampler(data, batch_size = 1000, num_workers = num_workers, walk_length =2, num_steps= 27, sample_coverage=10, save_dir="/home/ken/saint")
datasets = []
for d_idx , dataset in enumerate(train_loader_graphsaint):
    print("start:" , d_idx)
    datasets.append(dataset)

