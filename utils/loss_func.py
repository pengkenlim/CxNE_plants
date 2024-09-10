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

def calculate_cosine_sim(embeddings):
    embeddings_norm = F.normalize(embeddings, dim=1)
    cosine_similarities = torch.mm(embeddings_norm, embeddings_norm.t())
    return extract_top_half(cosine_similarities)

def calculate_dot_product(embeddings):
    dotproduct = torch.mm(embeddings, embeddings.t())
    return extract_top_half(dotproduct)

def subset_coexp_str(scaled_coexp_str_adj, node_indices):
    return extract_top_half(torch.tensor(scaled_coexp_str_adj[node_indices, :][:, node_indices]).float())


def RMSE_dotprod_vs_coexp(embeddings, node_indices, coexp_str_adj):
    """L = sqrt(Σ [(dot_prod(x_i, x_j) - w_ij)^2] / |E|)"""
    dot_prod_1D = calculate_dot_product(embeddings)
    subseted_coexp_str_1D = subset_coexp_str(coexp_str_adj, node_indices)
    squared_error = (dot_prod_1D - subseted_coexp_str_1D)**2
    RMSE = torch.sqrt(squared_error.nanmean())
    return RMSE
