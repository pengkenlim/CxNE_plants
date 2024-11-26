import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Flickr
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.typing import WITH_TORCH_SPARSE
from torch_geometric.utils import degree


path = "/mnt/md2/ken/CxNE_plants_data/data/Flickr"
dataset = Flickr(path)
data = dataset[0]
row, col = data.edge_index
in_degree = degree(col, data.num_nodes)[col]
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

parser = argparse.ArgumentParser()
parser.add_argument('--use_normalization', action='store_true')
args = parser.parse_args()

loader = GraphSAINTRandomWalkSampler(data, batch_size=6000, walk_length=2,
                                     num_steps=5, sample_coverage=100,
                                     save_dir=dataset.processed_dir,
                                     num_workers=4)


class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    model.set_aggr('add' if args.use_normalization else 'mean')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs


for epoch in range(1, 51):
    loss = train()
    accs = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
          f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')
    
#########################
import os
import sys
import numpy as np
import scipy.sparse as sp
import pickle
from torch_geometric.data import Data
import torch
import torch.nn.functional as F

from torch_geometric.datasets import Flickr
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.typing import WITH_TORCH_SPARSE
from torch_geometric.utils import degree

def adjacency_matrix_to_edge_index(adjacency_matrix, adjacency_matrix_2, normalize_in_degrees = False):
    coo_matrix = sp.coo_matrix(adjacency_matrix)
    row_indices = torch.tensor(coo_matrix.row, dtype=torch.long)
    col_indices = torch.tensor(coo_matrix.col, dtype=torch.long)
    edge_index = torch.stack([row_indices, col_indices], dim=0)
    edge_weights = torch.tensor(coo_matrix.data, dtype=torch.float)
    scaled_coexp_str = adjacency_matrix_2[row_indices, col_indices]
    scaled_coexp_str = torch.tensor(scaled_coexp_str, dtype=torch.float32)
    if not normalize_in_degrees:
        return edge_index, edge_weights, scaled_coexp_str
    else:
        return edge_index, edge_weights / adjacency_matrix.sum(axis = 1)[col_indices] , scaled_coexp_str


workdir_main = "/mnt/md2/ken/CxNE_plants_data/models"
workdir_root = "/mnt/md2/ken/CxNE_plants_data/"
species = "taxid59689"
edge_index = adjacency_matrix_to_edge_index(adj_mat_zscore_5percent, adj_mat_zscore_5percent_in_degrees)
node_degrees_to_normalize = adj_mat_zscore_5percent_in_degrees
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
adj_mat_zscore_5percent_path = os.path.join(workdir_root, species,"adj_mat_zscore_20percent.pkl")
with open(adj_mat_zscore_5percent_path, "rb") as f:
    adj_mat_zscore_5percent= pickle.load(f)

scaled_edge_weight_adj_mat_path = "/mnt/md2/ken/CxNE_plants_data/taxid3702/adj_mat_zscore_norm.pkl"
with open(scaled_edge_weight_adj_mat_path, "rb") as fin:
    scaled_edge_weight_adj_mat = pickle.load(fin)

#make data
scaled_edge_weight_adj_mat = adj_mat_zscore_5percent
edge_index, edge_weights, scaled_coexp_str = adjacency_matrix_to_edge_index(adj_mat_zscore_5percent, scaled_edge_weight_adj_mat,normalize_in_degrees = False)
data = Data(x=node_features, edge_index= edge_index, edge_weight = edge_weights)

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[node_split_idx["train_node_idx"]] = True

data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask[node_split_idx["val_node_idx"]] = True

data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[node_split_idx["test_node_idx"]] = True

#use y to store node idx so that we can fish out edge weights for contrastive loss function
data.y = torch.tensor(np.array([i for i in range(data.num_nodes)]))

data_path  = os.path.join(workdir_root, species,  "adj_mat_zscore_20percent_data.pkl")
with open(data_path, "wb") as fout:
    pickle.dump(data, fout)

########################################
#test GCN model
#####################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN, GAT
from torch_geometric.data import Data
import os
import sys
import numpy as np
import scipy.sparse as sp
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborLoader, NeighborSampler
import torch.nn.init as init
import pickle
from torch_geometric.utils import mask_to_index
import os
import sys
import numpy as np
import scipy.sparse as sp
import pickle
from torch_geometric.data import Data
import torch
import torch.nn.functional as F

from torch_geometric.datasets import Flickr
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.typing import WITH_TORCH_SPARSE
from torch_geometric.utils import degree

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


def train():
    model.train()
    model.set_aggr('add' if args.use_normalization else 'mean')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


#parameters
###########################
workdir_main = "/mnt/md2/ken/CxNE_plants_data/models"
workdir_root = "/mnt/md2/ken/CxNE_plants_data/"

##model
model_name = "model_GCN_saint_1"
dropout_rate = 0.01
hidden_channels = 480
num_layers = 3
out_channels = None
device = torch.device('cpu')
species = "taxid3702"
heads = 8

##training
##use adam optimizer by default
start_epoch = 1
end_epoch = 31 #exclude 31
learning_rate = 1e-3 #0.001
batch_size = 64
act = "relu"

#mini-batching
num_workers = 60 
batch_size = 250
walk_length=3 # equal to layers
sample_coverage = 1 # number of samplings to derive edge statistics
num_steps = 100 # mini-batches per epoch
save_dir = os.path.join(workdir_main, model_name + "_sampling_stats")

#???

#data
data_path = "/mnt/md2/ken/CxNE_plants_data/taxid3702/adj_mat_zscore_50percent_data_wonormindegrees.pkl"


#double check if there is nan
#np.where(np.isnan(scaled_edge_weight_adj_mat))

if not os.path.exists(workdir_main):
    os.makedirs(workdir_main)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("Initializing model and optimizer...")
model = GCN(in_channels = 480, 
            hidden_channels = hidden_channels, 
            num_layers  = num_layers, 
            out_channels  = out_channels, 
            act =act)
init_all_layers()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
logpath = os.path.join(workdir_main, model_name +".txt")

print("Loading data...")
with open(data_path, "rb") as fin:
    data = pickle.load(fin)



print("Creating minibatches...")
loader = GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=walk_length,
                                     num_steps=num_steps, sample_coverage=sample_coverage,
                                     #save_dir=save_dir,
                                     num_workers=num_workers)



subseted_scaled_coexp_str_adj = subset_scaled_coexp_str_adj(scaled_edge_weight_adj_mat, mini_batch.y)



def calculate_cosine_sim(embeddings):
    embeddings_norm = F.normalize(embeddings, dim=1)
    cosine_similarities = torch.mm(embeddings_norm, embeddings_norm.t())
    return cosine_similarities

def subset_scaled_coexp_str_adj(scaled_coexp_str_adj, node_indices):
    return torch.tensor(scaled_coexp_str_adj[node_indices, :][:, node_indices]).float()

def my_custom_loss_function_2(embeddings, subseted_scaled_coexp_str_adj):
    """L = sqrt(Σ [(cos(x_i, x_j) - w_ij)^2] / |E|)"""
    cosine_similarities = calculate_cosine_sim(embeddings)
    squared_error= (cosine_similarities - subseted_scaled_coexp_str_adj)**2
    RMSE = torch.sqrt((squared_error.sum())/ (len(embeddings)**2 - len(embeddings)) )
    return RMSE


#GCN
num_epochs = 100
epoch_train_performance = {}
model.train()
for epoch in range(num_epochs):
    total_summed_SE, total_num_contrasts = 0, 0
    mini_batches = [mini_batch for mini_batch in loader]
    for mb_idx , mini_batch in enumerate(mini_batches):
        mini_batch = mini_batch.to(device)
        optimizer.zero_grad()
        edge_weight = mini_batch.edge_norm * mini_batch.edge_weight.float() # graph saint normalization by sampling bias
        out = model(mini_batch.x, mini_batch.edge_index, edge_weight)
        train_out = out[mini_batch.train_mask]
        subseted_scaled_coexp_str_adj = subset_scaled_coexp_str_adj(scaled_edge_weight_adj_mat, mini_batch.y[mini_batch.train_mask])
        RMSE = my_custom_loss_function_2(train_out, subseted_scaled_coexp_str_adj)
        RMSE.backward()
        optimizer.step()
        num_contrasts = train_out.shape[0] **2
        RMSE = float(RMSE.detach())
        summed_SE = (RMSE**2)*num_contrasts
        total_num_contrasts += num_contrasts 
        total_summed_SE += summed_SE
        print(mb_idx, RMSE)
    epoch_train_performance[epoch] = {np.sqrt(total_summed_SE / total_num_contrasts)}
    print(f"epoch {epoch}, RMSE across batches: {epoch_train_performance[epoch]}")
    


#GAT
concat =True
print("Initializing model and optimizer...")
model = GAT(in_channels = 480, 
            hidden_channels = hidden_channels, 
            num_layers  = num_layers, 
            out_channels  = out_channels, 
            act =act,
            v2=True,
            concat = concat ,
            heads = heads,
            edge_dim =1,)

init_all_layers()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
logpath = os.path.join(workdir_main, model_name +".txt")

print("Creating minibatches...")
loader = GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=walk_length,
                                     num_steps=num_steps, sample_coverage=sample_coverage,
                                     #save_dir=save_dir,
                                     num_workers=num_workers)


#GAT
num_epochs = 100
epoch_train_performance = {}
model.train()
for epoch in range(num_epochs):
    total_summed_SE, total_num_contrasts = 0, 0
    mini_batches = [mini_batch for mini_batch in loader]
    for mb_idx , mini_batch in enumerate(mini_batches):
        mini_batch = mini_batch.to(device)
        optimizer.zero_grad() 
        #edge_weight = mini_batch.edge_norm * mini_batch.edge_weight.float() # graph saint normalization by sampling bias
        edge_weight = mini_batch.edge_weight.float() # zscore MR from tea-gcn
        out = model(mini_batch.x, mini_batch.edge_index, edge_attr = edge_weight)
        #out = model(mini_batch.x, mini_batch.edge_index) # no edge attributes
        train_out = out[mini_batch.train_mask]
        subseted_scaled_coexp_str_adj = subset_scaled_coexp_str_adj(scaled_edge_weight_adj_mat, mini_batch.y[mini_batch.train_mask])
        RMSE = my_custom_loss_function_2(train_out, subseted_scaled_coexp_str_adj)
        RMSE.backward()
        optimizer.step()
        num_contrasts = train_out.shape[0] **2
        RMSE = float(RMSE.detach())
        summed_SE = (RMSE**2)*num_contrasts
        total_num_contrasts += num_contrasts 
        total_summed_SE += summed_SE
        print(mb_idx, RMSE)
    epoch_train_performance[epoch] = {np.sqrt(total_summed_SE / total_num_contrasts)}
    print(f"epoch {epoch}, RMSE across batches: {epoch_train_performance[epoch]}")



def train():
    model.train()
    model.set_aggr('add' if args.use_normalization else 'mean')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


#nmini-batch training
##############

#mini-batching and dataloading
num_workers = 60 
mini_batch_size = 100
actual_batch_size = 1000
num_neighbors = [2000, 2000] # -1 means all meighbours will be selected
data_path = "/mnt/md2/ken/CxNE_plants_data/taxid3702/adj_mat_zscore_5percent_data.pkl"
#save_dir = os.path.join(workdir_main, model_name + "_sampling_stats")

print("Loading data...")
with open(data_path, "rb") as fin:
    data = pickle.load(fin)

#print("writing data...")
#with open(data_path, "wb") as fout:
#    pickle.dump(data, fout)

input_nodes_for_training  = mask_to_index(data.train_mask)
loader = NeighborSampler(data.edge_index ,node_idx  = input_nodes_for_training, 
                        sizes  = num_neighbors , 
                        num_workers = num_workers,
                        batch_size = mini_batch_size,
                        #weight_attr = "edge_weight", 
                        shuffle=True)

#loader = NeighborLoader(data,input_nodes  = input_nodes_for_training, 
#                        num_neighbors  = num_neighbors , 
#                        num_workers = num_workers,
#                        batch_size = mini_batch_size,
#                        weight_attr = "edge_weight", 
#                        shuffle=True)

mini_batch = next(iter(loader))
test_minibatch_path = "/mnt/md2/ken/CxNE_plants_data/taxid3702/test_minibatch.pkl"
with open(test_minibatch_path, "wb") as fout:
    pickle.dump(mini_batch, fout)

#Hybrid models
import os
import sys
import numpy as np
import scipy.sparse as sp
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN, GAT
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborLoader, NeighborSampler
import torch.nn.init as init
from torch_geometric.utils import mask_to_index, degree , trim_to_layer
from torch_geometric.datasets import Flickr
from torch_geometric.nn import GraphConv, SGConv, GATv2Conv
from torch_geometric.typing import WITH_TORCH_SPARSE


class SG_GATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, act, concat, heads, K, edge_dim= 1, act_kwargs= None, num_layers= 2):
        from  torch_geometric.nn.resolver import activation_resolver
        super(SG_GATv2, self).__init__()
        self.convs = nn.ModuleList()
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.num_layers = num_layers
        #for GAT layer
        self.concat = concat
        self.edge_dim  = edge_dim
        #for SGconv layer
        self.K  = K
        if type(hidden_channels) == int:
            first_layer = SGConv(in_channels, hidden_channels, K )
            self.convs.append(first_layer)
            second_layer = GATv2Conv(hidden_channels, hidden_channels, concat= concat, heads = heads, edge_dim =edge_dim)
            self.convs.append(second_layer)
        
        elif len(hidden_channels) > 1:
            first_layer = SGConv(in_channels, hidden_channels[0], K )
            self.convs.append(first_layer)
            second_layer = GATv2Conv(hidden_channels[0], hidden_channels[1], concat= concat, edge_dim =edge_dim)
            self.convs.append(second_layer)

    def forward(self, x, edge_index, edge_weight, layer_specific = None):
        if layer_specific==None:
            for i in range(len(self.convs)):
                if i < len(self.convs) - 1:
                    x = self.convs[i](x, edge_index, edge_weight = edge_weight)
                    x= self.act(x)
                else:
                    x = self.convs[i](x, edge_index, edge_attr  = edge_weight)
        else:
            assert layer_specific <  len(self.convs) +1
            if layer_specific  < len(self.convs):
                x = self.convs[layer_specific -1 ](x, edge_index, edge_weight = edge_weight)
                x= self.act(x)
            else:
                x = self.convs[layer_specific-1](x, edge_index, edge_attr  = edge_weight)
        return x
    

def init_all_layers():
    for i in range(0,model.num_layers):
        if hasattr(model.convs[i], 'lin_l') and model.convs[i].lin_l is not None:
            init.kaiming_uniform_(model.convs[i].lin_l.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(model.convs[i].lin_l, 'bias') and model.convs[i].lin_l.bias is not None:
                init.zeros_(model.convs[i].lin_l.bias)
        if hasattr(model.convs[i], 'lin_r') and model.convs[i].lin_r is not None:
            init.kaiming_uniform_(model.convs[i].lin_r.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(model.convs[i].lin_r, 'bias') and model.convs[i].lin_r.bias is not None:
               init.zeros_(model.convs[i].lin_r.bias)

def calculate_cosine_sim(embeddings):
    embeddings_norm = F.normalize(embeddings, dim=1)
    cosine_similarities = torch.mm(embeddings_norm, embeddings_norm.t())
    return cosine_similarities

def subset_scaled_coexp_str_adj(scaled_coexp_str_adj, node_indices):
    return torch.tensor(scaled_coexp_str_adj[node_indices, :][:, node_indices]).float()

def my_custom_loss_function_2(embeddings, subseted_scaled_coexp_str_adj):
    """L = sqrt(Σ [(cos(x_i, x_j) - w_ij)^2] / |E|)"""
    cosine_similarities = calculate_cosine_sim(embeddings)
    squared_error= (cosine_similarities - subseted_scaled_coexp_str_adj)**2
    RMSE = torch.sqrt((squared_error.sum())/ (len(embeddings)**2 - len(embeddings)) )
    return RMSE


print("Initializing model and optimizer...")

model = SG_GATv2(in_channels = 480, 
            hidden_channels = [256 , 128], 
            act ="leakyrelu",
            K= 3,
            concat = False ,
            heads = 4,
            edge_dim = 1)

data_path = "/mnt/md2/ken/CxNE_plants_data/taxid3702/adj_mat_zscore_5percent_data.pkl"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_all_layers()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


print("Loading data...")
with open(data_path, "rb") as fin:
    data = pickle.load(fin)

scaled_edge_weight_adj_mat_path = "/mnt/md2/ken/CxNE_plants_data/taxid3702/adj_mat_zscore_norm.pkl"
with open(scaled_edge_weight_adj_mat_path, "rb") as fin:
    scaled_edge_weight_adj_mat = pickle.load(fin)
scaled_edge_weight_adj_mat = torch.tensor(scaled_edge_weight_adj_mat)

workdir_main = "/mnt/md2/ken/CxNE_plants_data/models"
workdir_root = "/mnt/md2/ken/CxNE_plants_data/"

##model
model_name = "Hybrid_model_1"
save_dir = os.path.join(workdir_main, model_name + "_sampling_stats")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

subgraph_size = 10_000
accumulation_steps = 10
mini_batch_size = 100

print("Creating minibatches...")
loader = GraphSAINTRandomWalkSampler(data, batch_size=subgraph_size, walk_length= model.K +1,
                                     num_steps=round(data.x.shape[0] / subgraph_size), sample_coverage=1,
                                     save_dir=save_dir,
                                     num_workers=60)

#subgraph->batch->mini-batch accumulation
num_epochs = 100
epoch_train_performance = {}
model.train()
subgraphs = [subgraph for subgraph in loader]
for epoch in range(num_epochs):
    total_summed_SE, total_num_contrasts = 0, 0
    for s_idx , subgraph in enumerate(subgraphs):
        mini_batch_loader = NeighborLoader(subgraph, input_nodes  = mask_to_index(subgraph.train_mask), 
                        num_neighbors  = [2000] , 
                        num_workers = 60,
                        batch_size = mini_batch_size,
                        weight_attr = "edge_weight", 
                        shuffle=True)

        layer_1_out = model( subgraph.x, subgraph.edge_index, subgraph.edge_weight, layer_specific = 1)
        #neighborhood = next(iter(mini_batch_loader))# remove
        batch_layer_2_out = {"acummulated_outputs":torch.tensor([]), "corresponding_node_id":torch.tensor([])}
        for mb_idx, neighborhood in enumerate(mini_batch_loader):
            optimizer.zero_grad()
            neighborhood_format2 = trim_to_layer(0, neighborhood.num_sampled_nodes, neighborhood.num_sampled_edges,
                                                 subgraph.y[neighborhood.n_id], #id context is at subgraph level. y is id at dataset level
                                                 neighborhood.edge_index)
            
            mini_batch_layer_2_out = model( (layer_1_out[neighborhood.n_id], layer_1_out[neighborhood.n_id][:neighborhood.batch_size]), 
                                           neighborhood_format2[1],  #id context is at subgraph level
                                           subgraph.edge_weight[neighborhood.e_id], layer_specific=2)
            
            batch_layer_2_out["acummulated_outputs"] = torch.cat((batch_layer_2_out["acummulated_outputs"],mini_batch_layer_2_out), 0)
            batch_layer_2_out["corresponding_node_id"] = torch.cat((batch_layer_2_out["corresponding_node_id"],
                                                                    neighborhood_format2[0][: neighborhood.batch_size]),  # this is y (i.e. id at dataset level)
                                                                   0)
            print(mb_idx, "completed")
            if (mb_idx + 2) % accumulation_steps == 0  or (mb_idx+1) == len(mini_batch_loader):
                #calculate loss function
                subseted_scaled_coexp_str_adj = subset_scaled_coexp_str_adj(scaled_edge_weight_adj_mat, 
                                                                            batch_layer_2_out["corresponding_node_id"].int())
                RMSE = my_custom_loss_function_2(batch_layer_2_out["acummulated_outputs"], subseted_scaled_coexp_str_adj)
                RMSE.backward()
                optimizer.step()

                #for calculating loss
                num_contrasts = (batch_layer_2_out["acummulated_outputs"].shape[0] **2) - batch_layer_2_out["acummulated_outputs"].shape[0] 
                RMSE = float(RMSE.detach())
                total_num_contrasts += num_contrasts
                summed_SE = (RMSE**2)*num_contrasts
                total_summed_SE += summed_SE
                print(f"epoch: {epoch}, s_idx: {s_idx} , mb_idx: {mb_idx}, RMSE: {round(RMSE, 5)}")
                batch_layer_2_out = {"acummulated_outputs":torch.tensor([]), "corresponding_node_id":torch.tensor([])}
                layer_1_out = model( subgraph.x, subgraph.edge_index, subgraph.edge_weight, layer_specific = 1)
    epoch_train_performance[epoch] = {float(round(np.sqrt(total_summed_SE / total_num_contrasts) , 5))}
    print(f"epoch: {epoch} Epoch RMSE: {epoch_train_performance[epoch]}")




#loader = NeighborLoader(data,input_nodes  = input_nodes_for_training, 
#                        num_neighbors  = num_neighbors , 
#                        num_workers = num_workers,
#                        batch_size = mini_batch_size,
#                        weight_attr = "edge_weight", 
#                        shuffle=True)



#hybrid training
num_epochs = 100
epoch_train_performance = {}
model.train()
data.to(device)
for epoch in range(num_epochs):
    total_summed_SE, total_num_contrasts = 0, 0
    mini_batches = [mini_batch for mini_batch in loader]
    optimizer.zero_grad() 
    out = model(data.x, data.edge_index, data.edge_weight)
    RMSE = my_custom_loss_function_2(out, scaled_edge_weight_adj_mat)
    RMSE.backward()
    optimizer.step()
    print(epoch, np.float64(RMSE.detach()))
    epoch_train_performance[epoch] = np.float64(RMSE.detach())


        #subseted_scaled_coexp_str_adj = subset_scaled_coexp_str_adj(scaled_edge_weight_adj_mat, mini_batch.y[mini_batch.train_mask])
        
        RMSE.backward()
        optimizer.step()
        num_contrasts = train_out.shape[0] **2
        RMSE = float(RMSE.detach())
        summed_SE = (RMSE**2)*num_contrasts
        total_num_contrasts += num_contrasts 
        total_summed_SE += summed_SE
        print(mb_idx, RMSE)
    epoch_train_performance[epoch] = {np.sqrt(total_summed_SE / total_num_contrasts)}
    print(f"epoch {epoch}, RMSE across batches: {epoch_train_performance[epoch]}")

#hybrid graphsaint
num_epochs = 100
epoch_train_performance = {}
model.train()
for epoch in range(num_epochs):
    total_summed_SE, total_num_contrasts = 0, 0
    mini_batches = [mini_batch for mini_batch in loader]
    for mb_idx , mini_batch in enumerate(mini_batches):
        mini_batch = mini_batch.to(device)
        optimizer.zero_grad() 

        out = model(mini_batch.x, mini_batch.edge_index, mini_batch.edge_weight.float())

        train_out = out[mini_batch.train_mask]
        subseted_scaled_coexp_str_adj = subset_scaled_coexp_str_adj(scaled_edge_weight_adj_mat, mini_batch.y[mini_batch.train_mask])
        RMSE = my_custom_loss_function_2(train_out, subseted_scaled_coexp_str_adj)
        RMSE.backward()
        optimizer.step()
        num_contrasts = train_out.shape[0] **2
        RMSE = float(RMSE.detach())
        summed_SE = (RMSE**2)*num_contrasts
        total_num_contrasts += num_contrasts 
        total_summed_SE += summed_SE
        print(mb_idx, RMSE)
    epoch_train_performance[epoch] = {np.sqrt(total_summed_SE / total_num_contrasts)}
    print(f"epoch {epoch}, RMSE across batches: {epoch_train_performance[epoch]}")

#################
#inference-by-layer
##################
model = GAT(in_channels = 480, 
            hidden_channels = 256, 
            out_channels = None,
            v2 =True,
            act ="leakyrelu",
            num_layers = 3,
            concat = False,
            heads = 4,
            edge_dim = 1)

inference_batch_size = 1000
inference_batch_loader = NeighborLoader(data, 
                                   num_neighbors  = [2000] , 
                                   num_workers = 60,
                                   batch_size = inference_batch_size,
                                   weight_attr = "edge_weight", 
                                   shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_all_layers()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(num_epochs):
    #inference_batch = next(iter(inference_batch_loader)) # remove
    for i in range(len(model.convs)):
        layer_out = torch.tensor([])
        if i == 0:
            for ib_idx, inference_batch in enumerate(inference_batch_loader):
                edge_index = inference_batch.edge_index
                edge_weight= inference_batch.edge_weight
                x = inference_batch.x
                actual_inference_batch_size = inference_batch.batch_size
                out = model.act(model.convs[i]( (x, x[:actual_inference_batch_size]) ,
                               edge_index,
                               edge_weight = edge_weight))
                
                layer_out = torch.cat( (layer_out, out), 0 )
                print(ib_idx)

                #trim_to_layer(0,inference_batch.num_sampled_nodes, inference_batch.num_sampled_edges,
                #            inference_batch.x, inference_batch.edge_index, inference_batch.edge_weight)

class MLP_SG_GAT(nn.Module):
    def __init__(self, act="leakyrelu", act_kwargs= None):
        from  torch_geometric.nn.resolver import activation_resolver
        
        super(MLP_SG_GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.num_layers = 2
        self.act = activation_resolver(act, **(act_kwargs or {}))
        
        self.encode = nn.Sequential(
            nn.Linear(480, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            self.act)
        
        self.decode = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Linear(256, 480),
            nn.BatchNorm1d(480),
            self.act)
        self.K= 3
       
        first_layer = SGConv(128, 128, self.K)
        self.convs.append(first_layer)
        second_layer = GATv2Conv(128, 128, concat= False, heads = 4, edge_dim =-1)
        self.convs.append(second_layer)
        
        # Apply Kaiming initialization to all layers
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, SGConv) or isinstance(m, GATv2Conv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leakyrelu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x, edge_index, edge_weight):
        from  torch_geometric.nn import BatchNorm
        x= self.encode(x)
        x = self.convs[0](x, edge_index, edge_weight = edge_weight)
        x = BatchNorm(x.size(1))(x)
        x= self.act(x)
        x= self.convs[1](x, edge_index, edge_attr = edge_weight)
        x = BatchNorm(x.size(1))(x)
        x= self.act(x)
        x= self.decode(x)
        return x


class MLP_SG_GAT(nn.Module):
    def __init__(self, act="leakyrelu", act_kwargs= None):
        from  torch_geometric.nn.resolver import activation_resolver
        super(MLP_SG_GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.num_layers = 2
        self.act = activation_resolver(act, **(act_kwargs or {})) 
        self.encode = nn.Sequential(
            nn.Linear(480, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Linear(256, 128))
        for m in self.encode:
            if isinstance(m,  nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
        self.decode = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Linear(256, 256))
        for m in self.decode:
            if isinstance(m,  nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
        self.K= 2
        self.convs.append(SGConv(128, 128, self.K))
        self.convs.append(GATv2Conv(128, 128, concat= False, heads = 4, edge_dim =-1))
        self.convs.append(GATv2Conv(128, 128, concat= False, heads = 4, edge_dim =-1))
        for conv in self.convs:
            if isinstance(conv, SGConv):
                nn.init.kaiming_normal_(conv.lin.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(conv.lin.bias, 0)
            elif isinstance(conv, SGConv):
                nn.init.kaiming_normal_(conv.lin_l.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.kaiming_normal_(conv.lin_r.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(conv.lin_l.bias, 0)
                nn.init.constant_(conv.lin_r.bias, 0)
    def forward(self, x, edge_index, edge_weight):
        from  torch_geometric.nn import BatchNorm
        x= self.encode(x)
        for i, conv in enumerate(self.convs):
            if isinstance(conv, SGConv):
                x = conv(x, edge_index, edge_weight = edge_weight)
            elif isinstance(conv, GATv2Conv):
                x = conv(x, edge_index, edge_attr = edge_weight)
            if i < (len(self.convs) -1):
                x = self.act( BatchNorm(x.size(1))(x) )
        x= self.decode(x)
        return x


from torch_geometric.profile import count_parameters
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
model = MLP_SG_GAT()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

cluster_data = ClusterData(data, num_parts = 100 , keep_inter_cluster_edges =True)
Cluster_Loader = ClusterLoader(cluster_data, num_workers=60, batch_size= 20, shuffle=True)
#cluster = next(iter(Cluster_Loader)) # remove

print("Creating minibatches...")
loader = GraphSAINTRandomWalkSampler(data, batch_size=5000, walk_length= model.K +1,
                                     num_steps=round(data.x.shape[0] / 5000), sample_coverage=1,
                                     save_dir=save_dir,
                                     num_workers=60)

epoch_train_performance = {}
model.train()
for epoch in range(num_epochs):
    total_summed_SE, total_num_contrasts = 0, 0
    for mb_idx, cluster in enumerate(loader):
        optimizer.zero_grad()
        out = model(cluster.x,  cluster.edge_index, cluster.edge_weight)
        train_out = out[cluster.train_mask]
        subseted_scaled_coexp_str_adj = subset_scaled_coexp_str_adj(scaled_edge_weight_adj_mat, cluster.y[cluster.train_mask])
        RMSE = my_custom_loss_function_2(train_out, subseted_scaled_coexp_str_adj)
        RMSE.backward()
        optimizer.step()
        num_contrasts = train_out.shape[0] **2
        RMSE = float(RMSE.detach())
        summed_SE = (RMSE**2)*num_contrasts
        total_num_contrasts += num_contrasts 
        total_summed_SE += summed_SE
        print(mb_idx, RMSE)
    epoch_train_performance[epoch] = float(round(np.sqrt(total_summed_SE / total_num_contrasts), 5))
    print(f"epoch {epoch}, RMSE across batches: {epoch_train_performance[epoch]}")

# %%
# load
import torch
import os
import sys
import pickle
import numpy as np

tensor = torch.load('/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/interproscan/input_fasta/esm/taxid3702/ATMG01370.1 pacid=37383531 transcript=ATMG01370.1 locus=ATMG01370 ID=ATMG01370.1.Araport11.447 annot-version=Araport11.pt')

tensor["mean_representations"][36]
# %%
workdir = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/interproscan/input_fasta/esm"
species_data_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/"
species_list  = [species for species in os.listdir(workdir) if "taxid" in species]
species_list = [species for species in species_list if species != "taxid81970"]


# %%
embdims = 2560
layer_extracted = 36
species_list = ["taxid59689", "taxid3694", "taxid29760"]
for species in species_list:
    #load
    gene_dict_path = os.path.join(species_data_dir,species,"gene_dict.pkl")
    Tid2Gid_dict_path = os.path.join(species_data_dir,species,"Tid2Gid_dict.pkl")
    with open(gene_dict_path, "rb") as fbin:
        gene_dict = pickle.load(fbin)
    node_features = np.zeros((len(gene_dict),embdims ) , dtype="float16")
    with open(Tid2Gid_dict_path, "rb") as fbin:
        Tid2Gid_dict = pickle.load(fbin)
    emb_filenames = os.listdir(os.path.join(workdir, species))
    #print(species, "embs:",len(emb_filenames))
    print(species, "Genes:",len(gene_dict))
    n=0
    for emb_filename in emb_filenames:
        try:
            tid = emb_filename.split(" ")[0]
            gid = Tid2Gid_dict[tid]
            g_index = gene_dict[gid]
            #load emb
            emb = torch.load(os.path.join(workdir, species, emb_filename))["mean_representations"][layer_extracted].numpy().astype("float16")
            node_features[g_index,:] = emb
            n+= 1
            print(n)
        except:
            pass
    node_features_path = os.path.join(species_data_dir,species, "ESM3B_node_features.pkl")
    with open(node_features_path, "wb") as fbout:
        pickle.dump(node_features, fbout)
    print(species, "embs_added:",n)

# %%
import random
import math
import scipy.sparse as sp
#make data
def create_random_split(n_nodes, splits):
    # Normalize splits to sum to 1
    total_split = sum(splits)
    normalized_splits = [s / total_split for s in splits]

    # Calculate the number of nodes for each split
    split_sizes = [math.floor(n_nodes * s) for s in normalized_splits]

    # Adjust for rounding errors to ensure the total equals n_nodes
    while sum(split_sizes) < n_nodes:
        for i in range(len(split_sizes)):
            if sum(split_sizes) < n_nodes:
                split_sizes[i] += 1

    # Shuffle the node indices randomly
    all_nodes = list(range(n_nodes))
    random.shuffle(all_nodes)

    # Assign nodes to train, validation, and test
    train_node_idx = torch.tensor(all_nodes[:split_sizes[0]])
    val_node_idx = torch.tensor(all_nodes[split_sizes[0]:split_sizes[0] + split_sizes[1]])
    test_node_idx = torch.tensor(all_nodes[split_sizes[0] + split_sizes[1]:])

    return train_node_idx, val_node_idx, test_node_idx
def adjacency_matrix_to_edge_index(adjacency_matrix):
    coo_matrix = sp.coo_matrix(adjacency_matrix)
    row_indices = torch.tensor(coo_matrix.row, dtype=torch.long)
    col_indices = torch.tensor(coo_matrix.col, dtype=torch.long)
    edge_index = torch.stack([row_indices, col_indices], dim=0)
    edge_weights = torch.tensor(coo_matrix.data, dtype=torch.float)

    return edge_index, edge_weights

# %%
#load nodes_split
species = "taxid59689"
overwrite_node_split = True
#load genedict
gene_dict_path = os.path.join(species_data_dir, species, "gene_dict.pkl")
with open(gene_dict_path, "rb") as f:
    gene_dict= pickle.load(f)

#load node features
node_features_path = os.path.join(species_data_dir, species,  "ESM3B_node_features.pkl")
with open(node_features_path, "rb") as f:
    node_features= pickle.load(f)
node_features  = torch.tensor(node_features,dtype=torch.float16 )

node_split_idx_path = os.path.join(species_data_dir, species,"node_split_idx.pkl")
if overwrite_node_split:
    n_nodes = len(gene_dict)
    train_node_idx, val_node_idx, test_node_idx = create_random_split(n_nodes, [0.8,0.1,0.1])
    node_split_idx = {"train_node_idx" : train_node_idx,
                      "val_node_idx": val_node_idx,
                      "test_node_idx": test_node_idx }
    with open(node_split_idx_path, "wb") as fbout:
        pickle.dump(node_split_idx, fbout)
else:
    with open(node_split_idx_path, "rb") as f:
        node_split_idx= pickle.load(f)


#load adj
adj_mat_zscore_5percent_path = os.path.join(species_data_dir, species,"adj_mat_zscore_20percent.pkl")
with open(adj_mat_zscore_5percent_path, "rb") as f:
    adj_mat_zscore_5percent= pickle.load(f)


from torch_geometric.data import Data
edge_index, edge_weights = adjacency_matrix_to_edge_index(adj_mat_zscore_5percent)
data = Data(x=node_features, edge_index= edge_index, edge_weight = edge_weights.to(dtype=torch.float16))

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[node_split_idx["train_node_idx"]] = True

data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask[node_split_idx["val_node_idx"]] = True

data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[node_split_idx["test_node_idx"]] = True

#use y to store node idx so that we can fish out edge weights for contrastive loss function
data.y = torch.tensor(np.array([i for i in range(data.num_nodes)]))

data_path =os.path.join(species_data_dir, species,"adj_mat_zscore_20percent_ESM3B_data.pkl")
with open(data_path, "wb") as fbout:
    pickle.dump(data, fbout)
# %%
