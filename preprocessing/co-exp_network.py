#setting sys.path for importing modules
import os
import sys
import scipy
import math
import numpy as np

if __name__ == "__main__":
         abspath= __file__
         parent_module= "/".join(abspath.split("/")[:-2])
         sys.path.insert(0, parent_module)

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


def load_network( network_dir , gene_dict=None):
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
                    weight = float(line_contents[-1])
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
  """
  Calculates the z-score for a given percentile in a standard normal distribution.

  Args:
    percentile: The desired percentile (0-100).

  Returns:
    The corresponding z-score.
  """

  # Convert percentile to probability
  probability = percentile / 100

  # Calculate z-score
  z_score = scipy.stats.norm.ppf(probability)
  return z_score
    

import numpy as np
import pickle
import os

if __name__ == "__main__":
    outdir = "/mnt/md2/ken/CxNE_plants_data/taxid3702/"

    gene_dict_path = "/mnt/md2/ken/CxNE_plants_data/taxid3702/gene_dict.pkl"
    with open(gene_dict_path, "rb") as fin:
        gene_dict = pickle.load(fin)
    network_dir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b2/taxid3702/Add_ranks/combine_all_444k_RAvg"
    gene_dict, adj_mat = load_network( network_dir , gene_dict=gene_dict)
    
    gene_dict_flipped = {value : key for key, value in gene_dict.items()}
    #standardize
    adj_mat_zscore = zscore(-1* adj_mat)
    adj_mat_zscore[np.isnan(adj_mat_zscore)] = 0

    #save 
    adj_mat_zscore_path = os.path.join(outdir, "adj_mat_zscore.pkl")
    with open(adj_mat_zscore_path, "wb") as fout:
        pickle.dump( adj_mat_zscore, fout)

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
    density = 50
    cut_off = zscore_from_percentile(100- density) # zscore cutoff for 5% density
    adj_mat_zscore_5percent = adj_mat_zscore.copy()
    adj_mat_zscore_5percent[adj_mat_zscore_5percent < cut_off] = 0
    (adj_mat_zscore_5percent > 0).astype(int).sum()
    adj_mat_zscore_5percent_path = os.path.join(outdir, "adj_mat_zscore_50percent.pkl")
    with open(adj_mat_zscore_5percent_path, "wb") as fout:
        pickle.dump( adj_mat_zscore_5percent, fout)

    adj_mat_zscore_5percent_in_degrees = adj_mat_zscore_5percent.sum(axis = 1)
    adj_mat_zscore_5percent_in_degrees_path = os.path.join(outdir, "adj_mat_zscore_5percent_in_degrees.pkl")
    with open(adj_mat_zscore_5percent_in_degrees_path, "wb") as fout:
        pickle.dump( adj_mat_zscore_5percent_in_degrees, fout)

