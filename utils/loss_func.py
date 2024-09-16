#setting sys.path for importing modules
import os
import sys

if __name__ == "__main__":
         abspath= __file__
         parent_module= "/".join(abspath.split("/")[:-2])
         sys.path.insert(0, parent_module)

import torch.nn.functional as F
import torch


def extract_top_half(adjacency_matrix):
    """Extracts the top half of an adjacency matrix into a 1D array.

    Args:
        adjacency_matrix: A square adjacency matrix.

    Returns:
        A 1D array containing the upper triangular part of the adjacency matrix.
    """
    num_nodes = adjacency_matrix.size(0)
    indices = torch.triu_indices(num_nodes, num_nodes, offset=0)  # Get indices for the upper triangular part
    return adjacency_matrix[indices[0], indices[1]]

#def calculate_cosine_sim(embeddings):
#    embeddings_norm = F.normalize(embeddings, dim=1)
#    cosine_similarities = torch.mm(embeddings_norm, embeddings_norm.t())
#    return extract_top_half(cosine_similarities)

def calculate_dot_product(embeddings, target=None):
    if target == None:
        dotproduct = torch.mm(embeddings, embeddings.t())
    else:
        dotproduct = torch.mm(embeddings, target.t())
    return dotproduct

def subset_coexp_str(scaled_coexp_str_adj, node_indices, target_indices=None):
    if target_indices == None:
        return torch.tensor(scaled_coexp_str_adj[node_indices, :][:, node_indices]).float()
    else:
        return torch.tensor(scaled_coexp_str_adj[node_indices, :][:, target_indices]).float()


def RMSE_dotprod_vs_coexp(embeddings, node_indices, coexp_str_adj):
    """L = sqrt(Σ [(dot_prod(x_i, x_j) - w_ij)^2] / |E|)"""
    dot_prod_1D = extract_top_half(calculate_dot_product(embeddings))
    subseted_coexp_str_1D = extract_top_half(subset_coexp_str(coexp_str_adj, node_indices))
    squared_error = (dot_prod_1D - subseted_coexp_str_1D)**2
    RMSE = torch.sqrt(squared_error.nanmean())
    return RMSE


def RMSE_dotprod_vs_coexp_testval(embeddings_testval, node_indices_testval, embeddings_training , node_indices_training, coexp_adj_mat):
    """L = sqrt(Σ [(dot_prod(x_i, x_j) - w_ij)^2] / |E|)
    val v val nodes concat val v training nodes
    OR
    test v test nodes concat test v training nodes"""
    dot_prod_1D_testval = extract_top_half(calculate_dot_product(embeddings_testval))
    subseted_coexp_str_1D_testval = extract_top_half(subset_coexp_str(coexp_adj_mat, node_indices_testval))
    squared_error_testval = (dot_prod_1D_testval - subseted_coexp_str_1D_testval)**2

    dot_prod_testval_v_training = calculate_dot_product(embeddings_testval, target=embeddings_training)
    subseted_coexp_str_testval_v_training = subset_coexp_str(coexp_adj_mat, node_indices_testval, target_indices= node_indices_training)
    squared_error_testval_v_training = (dot_prod_testval_v_training - subseted_coexp_str_testval_v_training)**2

    num_comparisons = len(squared_error_testval) + (squared_error_testval_v_training.shape[0] * squared_error_testval_v_training.shape[1])
    RMSE_testval = torch.sqrt( (squared_error_testval_v_training.sum() + squared_error_testval.sum() ) / num_comparisons ) 
    return RMSE_testval, num_comparisons
