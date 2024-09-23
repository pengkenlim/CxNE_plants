import os
import sys



    
import torch
import numpy as np
import pickle
from scipy import stats, integrate
from sklearn import metrics
import torch.nn as nn


def calc_AUC_PRC(pos_scores, neg_scores, return_thresholds =False):
    scores = np.concatenate((pos_scores , neg_scores))
    labels = np.concatenate((np.ones(len(pos_scores)) , np.zeros(len(neg_scores)))) 
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=1)
    AUC_PRC = float(np.abs(integrate.trapezoid(y=precision, x=recall)))
    if return_thresholds:
        return precision, recall, thresholds, AUC_PRC
    else:
        return AUC_PRC


def extract_positive_scores(gene_dict, positive_edges, adj_mat ):
    pos_scores = []
    for edge in positive_edges:
        source , target = edge.split("--")
        s_idx , t_idx = gene_dict[source] , gene_dict[target]
        pos_scores.append(float(adj_mat[s_idx, t_idx]))
    return pos_scores

def calc_AUC_ROC(pos_score, neg_score, return_thresholds =False):
    scores = pos_score + neg_score
    labels = [1 for i in pos_score]    + [0 for i in neg_score]
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    AUC_ROC = float(metrics.auc(fpr, tpr))
    if return_thresholds:
        return fpr, tpr, thresholds, AUC_ROC
    else:
        return AUC_ROC

def calc_quartiles(performance_scores):
    return_array = list(stats.scoreatpercentile(performance_scores, [25,50,75], interpolation_method = "lower"))
    return [float(i) for i in return_array]

path_to_embedding = "/mnt/md2/ken/CxNE_plants_data/model_22_relu_loss_05psubgraph_10density/infered_emmbeddings/Epoch200_emb.pkl"

annot_dir = "/home/ken/Plant-GCN/test_data/ATTED_ARA_edges_GOfixed"
positive_met_edges_path = os.path.join(annot_dir, "positive_met_edges.pkl")
negative_met_edges_path = os.path.join(annot_dir, "negative_met_edges.pkl")
positive_TF_edges_path = os.path.join(annot_dir, "positive_TF_edges.pkl")
negative_TF_edges_path = os.path.join(annot_dir, "negative_TF_edges.pkl")
positive_GO_edges_path = os.path.join(annot_dir, "positive_GO_edges.pkl")
negative_GO_edges_path = os.path.join(annot_dir, "negative_GO_edges.pkl")

gene_dict_path = "/mnt/md2/ken/CxNE_plants_data/taxid3702/gene_dict.pkl"

coexp_adj_mat = '/mnt/md2/ken/CxNE_plants_data/taxid3702/adj_mat_zscore.pkl' 

#load edges
with open(positive_met_edges_path ,"rb") as fbin:
    positive_met_edges = pickle.load(fbin)

with open(positive_TF_edges_path ,"rb") as fbin:
    positive_TF_edges = pickle.load(fbin)

with open(positive_GO_edges_path ,"rb") as fbin:
    positive_GO_edges = pickle.load(fbin) 

with open(negative_met_edges_path ,"rb") as fbin:
    negative_met_edges = pickle.load(fbin) 
    

with open(negative_TF_edges_path ,"rb") as fbin:
    negative_TF_edges = pickle.load(fbin) 
    
with open(negative_GO_edges_path ,"rb") as fbin:
    negative_GO_edges = pickle.load(fbin) 



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

adj_mat = adj_mat.astype("float32")

#calculate AUPRC
#extract edges



def evaluate_ara_FULL(gene_dict, adj_mat, positive_met_edges, negative_met_edges,positive_GO_edges, negative_GO_edges, positive_TF_edges, negative_TF_edges):
    """main function. For evaluation of FULL networks"""
    performance_dict={}
    for edge_dataset in [ "Met", "GO", "TF"]:
        performance_dict[edge_dataset] = {"AUC_ROC":{},"AUC_PRC":{}, "AVG":{}}
        if edge_dataset == "Met":
            positive_edges , negative_edges = positive_met_edges, negative_met_edges
        elif edge_dataset == "GO":
            positive_edges , negative_edges = positive_GO_edges, negative_GO_edges
        elif edge_dataset == "TF":
            positive_edges , negative_edges = positive_TF_edges, negative_TF_edges
        positive_scores = extract_positive_scores(gene_dict, positive_edges, adj_mat )
        for ds , neg_edges  in negative_edges.items():
            negative_scores = extract_positive_scores(gene_dict, neg_edges, adj_mat )
            performance_dict[edge_dataset]["AUC_ROC"][ds] = float(calc_AUC_ROC(positive_scores, negative_scores, return_thresholds =False))
            performance_dict[edge_dataset]["AUC_PRC"][ds] = float(calc_AUC_PRC(positive_scores, negative_scores, return_thresholds =False))
            performance_dict[edge_dataset]["AVG"][ds] = float(np.mean([performance_dict[edge_dataset]["AUC_ROC"][ds], 
                                                                                 performance_dict[edge_dataset]["AUC_PRC"][ds]]))
            performance_dict[edge_dataset]["Quartiles"] = {}
            performance_dict[edge_dataset]["Thresholds"] = {}
        #AUC_ROC
        performance_dict[edge_dataset]["Quartiles"]["AUC_ROC"] = calc_quartiles(list(performance_dict[edge_dataset]["AUC_ROC"].values()))
        med_idx = list(performance_dict[edge_dataset]["AUC_ROC"].values()).index(performance_dict[edge_dataset]["Quartiles"]["AUC_ROC"][1])
        negative_scores = negative_scores = extract_positive_scores(gene_dict, negative_edges[med_idx], adj_mat )
        performance_dict[edge_dataset]["Thresholds"]["AUC_ROC"] = calc_AUC_ROC(positive_scores, negative_scores, return_thresholds =True)
        #AUC_PRC
        performance_dict[edge_dataset]["Quartiles"]["AUC_PRC"] = calc_quartiles(list(performance_dict[edge_dataset]["AUC_PRC"].values()))
        med_idx = list(performance_dict[edge_dataset]["AUC_PRC"].values()).index(performance_dict[edge_dataset]["Quartiles"]["AUC_PRC"][1])
        performance_dict[edge_dataset]["Thresholds"]["AUC_PRC"] = calc_AUC_PRC(positive_scores, negative_scores, return_thresholds =True)
        #AVG
        performance_dict[edge_dataset]["Quartiles"]["AVG"] = calc_quartiles(list(performance_dict[edge_dataset]["AVG"].values()))
        #harmonic mean of scores
    performance_dict["HM"] = {"AUC_ROC":{},"AUC_PRC":{}, "AVG":{}}
    for ds in negative_edges.keys():
        performance_dict["HM"]["AUC_ROC"][ds] = float(stats.hmean([performance_dict[edge_dataset]["AUC_ROC"][ds] for edge_dataset in ["Met", "GO", "TF"]]))
        performance_dict["HM"]["AUC_PRC"][ds] = float(stats.hmean([performance_dict[edge_dataset]["AUC_PRC"][ds] for edge_dataset in ["Met", "GO", "TF"]]))
        performance_dict["HM"]["AVG"][ds] = float(np.mean([performance_dict["HM"]["AUC_ROC"][ds], 
                                                                     performance_dict["HM"]["AUC_PRC"][ds]])   )        
        performance_dict["HM"]["Quartiles"] = {}
        performance_dict["HM"]["Quartiles"]["AUC_ROC"] = calc_quartiles(list(performance_dict["HM"]["AUC_ROC"].values()))
        performance_dict["HM"]["Quartiles"]["AUC_PRC"] = calc_quartiles(list(performance_dict["HM"]["AUC_PRC"].values()))
        performance_dict["HM"]["Quartiles"]["AVG"] = calc_quartiles(list(performance_dict["HM"]["AVG"].values()))          
    return performance_dict 

performance_dict_TEA = evaluate_ara_FULL(gene_dict, adj_mat, positive_met_edges, negative_met_edges,positive_GO_edges, negative_GO_edges, positive_TF_edges, negative_TF_edges)

def print_stats(performance_dict_TEA):
    print("HM ROC:",float(performance_dict_TEA["HM"]["Quartiles"]["AUC_ROC"][1]))
    print("HM PRC:",float(performance_dict_TEA["HM"]["Quartiles"]["AUC_PRC"][1]))
    print("Met ROC:",float(performance_dict_TEA["Met"]["Quartiles"]["AUC_ROC"][1]))
    print("Met PRC:",float(performance_dict_TEA["Met"]["Quartiles"]["AUC_PRC"][1]))
    print("TF ROC:",float(performance_dict_TEA["TF"]["Quartiles"]["AUC_ROC"][1]))
    print("TF PRC:",float(performance_dict_TEA["TF"]["Quartiles"]["AUC_PRC"][1]))
    print("GO ROC:",float(performance_dict_TEA["GO"]["Quartiles"]["AUC_ROC"][1]))
    print("GO PRC:",float(performance_dict_TEA["GO"]["Quartiles"]["AUC_PRC"][1]))

print_stats(performance_dict_TEA)
performance_dict_TEA_path = "/mnt/md2/ken/CxNE_plants_data/taxid3702/performance_dict_TEA.pkl"
print_stats(performance_dict_TEA)
with open(performance_dict_TEA_path, "wb") as fbout:
    pickle.dump(performance_dict_TEA,fbout)


performance_dict_dp = evaluate_ara_FULL(gene_dict, dot_prod_np, positive_met_edges, negative_met_edges,positive_GO_edges, negative_GO_edges, positive_TF_edges, negative_TF_edges)
print_stats(performance_dict_dp)
performance_dict_dp_path = path_to_embedding.replace(".pkl","_performance_dict_dp.pkl")
with open(performance_dict_dp_path, "wb") as fbout:
    pickle.dump(performance_dict_dp,fbout)

performance_dict_cs = evaluate_ara_FULL(gene_dict, cosine_np, positive_met_edges, negative_met_edges,positive_GO_edges, negative_GO_edges, positive_TF_edges, negative_TF_edges)
print_stats(performance_dict_cs)

performance_dict_cs_path = path_to_embedding.replace(".pkl","_performance_dict_cs.pkl")
with open(performance_dict_cs_path, "wb") as fbout:
    pickle.dump(performance_dict_cs,fbout )