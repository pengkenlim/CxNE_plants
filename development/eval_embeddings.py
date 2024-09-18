import os
import sys

import torch
import numpy as np
import pickle
from scipy import stats, integrate
from sklearn import metrics
import torch.nn as nn



path_to_embedding = "/workspace/data/model_19_relu_loss_05psubgraph_25density/infered_emmbeddings/Epoch200_emb.pkl"
annot_dir = "/workspace/data/taxid3702/annot"
met_edges_path = os.path.join(annot_dir, "positive_met_edges.pkl")
gene_dict_path = os.path.join(annot_dir, "gene_dict.pkl")

coexp_adj_mat = '/workspace/data/taxid3702/adj_mat_zscore.pkl' 

#load edges
with open(met_edges_path ,"rb") as fbin:
    positive_met_edges = pickle.load(fbin) 

#load gene_dict
with open(gene_dict_path, "rb") as fbin:
    gene_dict = pickle.load(fbin)

#load embeddings
with open(path_to_embedding, "rb") as fbin:
    embeddings = pickle.load(fbin)

dot_prod = torch.mm(embeddings, embeddings.t())
dot_prod_np = dot_prod.numpy()

embeddings_norm = nn.functional.normalize(embeddings, dim=1)
cosine_np = torch.mm(embeddings_norm, embeddings_norm.t()).numpy()


#load adj mat
with open(coexp_adj_mat, "rb") as fbin:
    adj_mat  = pickle.load(fbin)

#calculate AUPRC
#extract edges
def calc_AUC_PRC(pos_scores, neg_scores, return_thresholds =False):
    scores = np.concatenate((pos_scores , neg_scores))
    labels = np.concatenate((np.ones(len(pos_scores)) , np.zeros(len(neg_scores)))) 
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=1)
    AUC_PRC = np.abs(integrate.trapezoid(y=precision, x=recall))
    if return_thresholds:
        return precision, recall, thresholds, AUC_PRC
    else:
        return AUC_PRC

def extract_positive_scores(gene_dict, positive_edges, adj_mat ):
    pos_scores = []
    avoid_indices = []
    for edge in positive_edges:
        source , target = edge.split("--")
        s_idx , t_idx = gene_dict[source] , gene_dict[target]
        pos_scores.append(float(adj_mat[s_idx, t_idx ]))
        avoid_indices.append([s_idx,t_idx])
        avoid_indices.append([t_idx,s_idx])
    pos_scores = np.array(pos_scores)
    avoid_indices = np.array(avoid_indices)
    upper_tri_indices = np.triu_indices_from(adj_mat, k=1)
    avoid_rows = avoid_indices[:, 0]
    avoid_cols = avoid_indices[:, 1]
    avoid_mask = ~((np.isin(upper_tri_indices[0], avoid_rows)) & (np.isin(upper_tri_indices[1], avoid_cols)))
    neg_scores = adj_mat[upper_tri_indices][avoid_mask]
    return pos_scores, neg_scores


pos_scores, neg_scores = extract_positive_scores(gene_dict, positive_met_edges, cosine_np )
neg_scores = np.random.choice(neg_scores, 7211, replace=False)


AUC_PRC =  calc_AUC_PRC(pos_scores, np.random.choice(neg_scores, 7211, replace=False))
print(AUC_PRC)