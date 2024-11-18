# %%
#setting sys.path for importing modules
import os
import sys
import scipy
import math
import numpy as np
# %%
if __name__ == "__main__":
         abspath= __file__
         parent_module= "/".join(abspath.split("/")[:-2])
         sys.path.insert(0, parent_module)
# %%
def calculate_edge_index(s, t, num_nodes):
    if s > t:
        s, t = t, s  # Ensure s < t
        return False, num_nodes * s - ((s + 1) * s) // 2 + t - s - 1
    else:
        return True, num_nodes * s - ((s + 1) * s) // 2 + t - s - 1

def init_flat_half_adj(num_edges):
    edge_values = np.empty(num_edges)
    edge_values[:] = math.nan
    return edge_values

def init_adj(num_genes):
    edge_values = np.empty( (num_genes, num_genes) )
    edge_values[:] = math.nan
    return edge_values


def load_network( network_dir , gene_dict=None, col_idx = 4):
    if gene_dict == None:
        genes = os.listdir(network_dir)
        gene_dict = {gene:i for i, gene in enumerate(genes)}
    else:
        genes = list(gene_dict.keys())
    adj_mat = init_adj(len(genes))
    for idx, source in enumerate(genes):
        with open(os.path.join(network_dir, source), "r") as f:
            for line_no, line in enumerate(f):
                if line_no != 0 and line != "": #skip first and last line
                    line_contents = line.split("\n")[0].split("\t")
                    target =  line_contents[0]
                    weight = float(line_contents[col_idx])
                    source_idx = gene_dict[source]
                    target_idx = gene_dict[target]
                    adj_mat[source_idx][target_idx] = weight
                    adj_mat[target_idx][source_idx] = weight
        if idx % 1000 ==0:
            print(idx,"genes loaded")
    return gene_dict, adj_mat

def zscore(data):
  mean = np.nanmean(data)
  std = np.nanstd(data)
  return (data - mean) / std


def zscore_from_percentile(percentile):
  # Convert percentile to probability
  probability = percentile / 100
  # Calculate z-score
  z_score = scipy.stats.norm.ppf(probability)
  return z_score
    

import numpy as np
import pickle
import os
# %%
#if __name__ == "__main__":
# %% 
#outdir = "/mnt/md2/ken/CxNE_plants_data/species_data/taxid29760/"
#if not os.path.exists(outdir):
#    os.makedirs(outdir)
# %% 
#gene_dict_path = "/mnt/md2/ken/CxNE_plants_data/taxid3702/gene_dict.pkl"
#with open(gene_dict_path, "rb") as fin:
#    gene_dict = pickle.load(fin)
# %%
#network_dir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b2/taxid3711/Build_ensemble_GCN/PCC/PCC_1k_Avg/" 
#genes = os.listdir(network_dir)
#with open(os.path.join(network_dir, genes[0]), 'r') as fin:
#    contents = fin.read()
#contents = contents.split("\n")
#gene_dict = {line.split("\t")[0] : idx for idx , line in enumerate(contents[1:-1])}
# %% 
#outdir = "/mnt/md2/ken/CxNE_plants_data/species_data/taxid29760/"
#network_dir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b1/taxid29760/Build_ensemble_GCN/PCC/PCC_1k_Avg"
if False:
    outdirs = ["/mnt/md2/ken/CxNE_plants_data/species_data/taxid4755/",
            "/mnt/md2/ken/CxNE_plants_data/species_data/taxid3880/",
            "/mnt/md2/ken/CxNE_plants_data/species_data/taxid4081/",
            "/mnt/md2/ken/CxNE_plants_data/species_data/taxid39947/"]
    network_dirs =["/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b1/taxid4577/Build_ensemble_GCN/PCC/PCC_1k_Avg",
                "/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b1/taxid3880/Build_ensemble_GCN/PCC/PCC_1k_Avg",
                "/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b1/taxid4081/Build_ensemble_GCN/PCC/PCC_1k_Avg",
                "/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b2/taxid39947/Build_ensemble_GCN/PCC/PCC_1k_Avg"]
    for outdir, network_dir in zip(outdirs, network_dirs):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        gene_dict, adj_mat = load_network(network_dir , gene_dict=None, col_idx = -2)

        print(f"loaded {network_dir}")
        #standardize
        adj_mat_zscore = zscore(adj_mat) # negative 1 for MR
        adj_mat_zscore[np.isnan(adj_mat_zscore)] = 0
        print("Calculated z score")

        #convert to float 16
        adj_mat_zscore = adj_mat_zscore.astype("float16")
        print("convert adj_mat_zscore to float 16")
        #save 
        adj_mat_zscore_path = os.path.join(outdir, "adj_mat_zscore_PCC_k1_AVG.pkl")
        with open(adj_mat_zscore_path, "wb") as fout:
            pickle.dump( adj_mat_zscore, fout)
        print("saved adj_mat_zscore")

        gene_dict_path = os.path.join(outdir, "gene_dict.pkl")
        with open(gene_dict_path, "wb") as fin:
                pickle.dump(gene_dict, fin)
        print("saved gene_dict")

###########################
outdir = "/mnt/md2/ken/CxNE_plants_data/species_data/taxid4577/"
gene_dict_path = os.path.join(outdir,"gene_dict.pkl" )
with open(gene_dict_path, "rb") as fin:
    gene_dict = pickle.load(fin)
network_dir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b1/taxid4577/Add_ranks/combine_all_562k_RAvg"
gene_dict, adj_mat = load_network(network_dir , gene_dict=gene_dict, col_idx = 3)
print(f"loaded {network_dir}")


#standardize
adj_mat_zscore = zscore(-1* adj_mat) # negative 1 for MR
adj_mat_zscore[np.isnan(adj_mat_zscore)] = 0
print("Calculated z score")

#convert to float 16
adj_mat_zscore = adj_mat_zscore.astype("float16")
print("convert adj_mat_zscore to float 16")

#save 
adj_mat_zscore_path = os.path.join(outdir, "adj_mat_zscore.pkl")
with open(adj_mat_zscore_path, "wb") as fout:
    pickle.dump( adj_mat_zscore, fout)
# %% 
sys.exit()

# %% 
#load
adj_mat_zscore_path = os.path.join(outdir, "adj_mat_zscore.pkl")
with open(adj_mat_zscore_path, "wb") as fout:
        pickle.dump( adj_mat_zscore, fout)

adj_mat_zscore_norm = 2 * (adj_mat_zscore - np.nanmin(adj_mat_zscore)) / (np.nanmax(adj_mat_zscore) - np.nanmin(adj_mat_zscore) ) - 1

adj_mat_zscore_norm_path = os.path.join(outdir, "adj_mat_zscore_norm.pkl")
with open(adj_mat_zscore_norm_path, "wb") as fout:
        pickle.dump( adj_mat_zscore_norm, fout)

with open(adj_mat_zscore_norm_path, "rb") as fin:
        adj_mat_zscore_norm = pickle.load(fin)
    
adj_mat_zscore_norm_ranks = scipy.stats.rankdata(-1 * adj_mat_zscore_norm, axis = 1, nan_policy = "omit") #rank by rows
    
adj_mat_zscore_norm_ranks_path = os.path.join(outdir, "adj_mat_zscore_norm_ranks.pkl")
with open(adj_mat_zscore_norm_ranks_path, "wb") as fout:
        pickle.dump( adj_mat_zscore_norm_ranks, fout)
    
    #binarize only rank >1000
adj_mat_zscore_norm_ranks_bin1000 = (adj_mat_zscore_norm_ranks <= 1000).astype(int)

    #ranks are assymetrical edge might be included 
adj_mat_zscore_norm_ranks_bin1000
adj_mat_zscore_norm_ranks_bin1000_symmetrical = np.maximum(adj_mat_zscore_norm_ranks_bin1000, adj_mat_zscore_norm_ranks_bin1000.T)
num_edges = (adj_mat_zscore_norm_ranks_bin1000_symmetrical.sum() - adj_mat_zscore_norm_ranks_bin1000_symmetrical.shape[0])/2

adj_mat_zscore_norm_ranks_bin1000_symmetrical_path = os.path.join(outdir, "adj_mat_zscore_norm_ranks_bin1000_symmetrical.pkl")
with open(adj_mat_zscore_norm_ranks_bin1000_symmetrical_path, "wb") as fout:
    pickle.dump( adj_mat_zscore_norm_ranks, fout)



    # find cutoff
density = 20
cut_off = zscore_from_percentile(100- density) # zscore cutoff for 5% density
adj_mat_zscore_5percent = adj_mat_zscore.copy()
adj_mat_zscore_5percent[adj_mat_zscore_5percent < cut_off] = 0
(adj_mat_zscore_5percent > 0).astype(int).sum()
adj_mat_zscore_5percent_path = os.path.join(outdir, "adj_mat_zscore_20percent.pkl")
with open(adj_mat_zscore_5percent_path, "wb") as fout:
        pickle.dump( adj_mat_zscore_5percent, fout)

adj_mat_zscore_5percent_in_degrees = adj_mat_zscore_5percent.sum(axis = 1)
adj_mat_zscore_5percent_in_degrees_path = os.path.join(outdir, "adj_mat_zscore_5percent_in_degrees.pkl")
with open(adj_mat_zscore_5percent_in_degrees_path, "wb") as fout:
        pickle.dump( adj_mat_zscore_5percent_in_degrees, fout)

#/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b1/taxid29760/Add_ranks/combine_all_375k_RAvg/#
#/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b1/taxid29760/Build_ensemble_GCN/PCC/PCC_1k_Avg#

#%%
#convert 

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
#%%
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

#%%

workdir_root = "/mnt/md2/ken/CxNE_plants_data/species_data"
species = "taxid3702"
species_old = "taxid3702_old"
#%%
#load adj_mat_zscore
adj_mat_zscore_path = os.path.join(workdir_root,species_old, "adj_mat_zscore.pkl")
with open(adj_mat_zscore_path, "rb") as fbin:
     adj_mat_zscore = pickle.load(fbin)
#%%
adj_mat_zscore = adj_mat_zscore.astype("float16")
#%%
if not os.path.exists(os.path.join(workdir_root,species) ):
     os.makedirs(os.path.join(workdir_root,species))

adj_mat_zscore_path = os.path.join(workdir_root,species, "adj_mat_zscore.pkl")

with open(adj_mat_zscore_path, "wb") as fbout:
    pickle.dump(adj_mat_zscore, fbout)


#%%
workdir_root = "/mnt/md2/ken/CxNE_plants_data/species_data"
species = "taxid4577"
adj_mat_zscore_path = os.path.join(workdir_root,species, "adj_mat_zscore.pkl")
with open(adj_mat_zscore_path, "rb") as fbin:
     adj_mat_zscore = pickle.load(fbin)

density = 20
cut_off = zscore_from_percentile(100- density) # zscore cutoff for 5% density

adj_mat_zscore_5percent = adj_mat_zscore.copy()
adj_mat_zscore_5percent[adj_mat_zscore_5percent < cut_off] = 0
(adj_mat_zscore_5percent > 0).astype(int).sum()

adj_mat_zscore_5percent_path = os.path.join(workdir_root,species, "adj_mat_zscore_20percent.pkl")
with open(adj_mat_zscore_5percent_path, "wb") as fout:
        pickle.dump( adj_mat_zscore_5percent, fout)




#%%
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