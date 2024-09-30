# %%
import os
import sys

import pandas as pd
import numpy as np

def edit_geneid(geneid):
    return geneid.split(".")[0].upper()

def return_1v1_ortholog(orthologues_dir):
    Ortholog_dict = {}
    for sp in species:
        sp_dir = os.path.join(orthologues_dir, f"Orthologues_{sp}")
        Ortholog_dict[sp] ={}
        for file in os.listdir(sp_dir):
            sp_target = file.split("__v__")[-1].replace(".tsv", "")
            Ortholog_dict[sp][sp_target] = {}
            with open(os.path.join(sp_dir, file), "r") as fin:
                contents = fin.read().split("\n")
            overlap = 0
            for line in contents[1:]:
                if "," not in line and line != "":
                    source = edit_geneid(line.split("\t")[1])
                    target = edit_geneid(line.split("\t")[-1])
                    Ortholog_dict[sp][sp_target][source] = target
                    overlap += 1
            Ortholog_dict[sp][sp_target]["overlap"] = overlap
    return Ortholog_dict

#extract orthofinder
OF_res_dir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/Orthofinder/input_fasta/OrthoFinder/Results_Jan22_3"
orthologues_dir = os.path.join(OF_res_dir , "Orthologues")
species = [folder.replace("Orthologues_", "") for folder in os.listdir(orthologues_dir) if "." not in folder]


Ortholog_dict = return_1v1_ortholog(orthologues_dir)



#Calculate jaccard index
#get total number of genes first
def get_total_genes(OF_res_dir):
    Statistics_PerSpecies_path = os.path.join(OF_res_dir , "Comparative_Genomics_Statistics", "Statistics_PerSpecies.tsv")
    with open(Statistics_PerSpecies_path, "r") as fin:
        contents = fin.read().split("\n")
    species_total_genes_dict = {species: int(total) for species, total in zip(contents[0].split("\t")[1:],contents[1].split("\t")[1:])}
    return species_total_genes_dict

species_total_genes_dict = get_total_genes(OF_res_dir)

#make dataframe
def JI_dataframe(Ortholog_dict, species_total_genes_dict, order=None):
    if order == None:
        order = list(Ortholog_dict.keys())
    JI_array = np.zeros((len(order), len(order)), dtype="float32")
    gene_no_array =  np.zeros((len(order), len(order)), dtype="int")
    ortholog_no_array =  np.zeros((len(order), len(order)), dtype="int")
    for sp_idx , species in enumerate(order):
        #update gene_no_array
        gene_no_array[sp_idx, sp_idx] = species_total_genes_dict[species]
        for sp_idx_T , species_T in enumerate(order):
            if sp_idx_T > sp_idx:
                intersection = Ortholog_dict[species][species_T]["overlap"]
                union = (species_total_genes_dict[species] + species_total_genes_dict[species_T]) - intersection
                JI_array[sp_idx, sp_idx_T] = intersection / union
            elif sp_idx_T < sp_idx:
                intersection = Ortholog_dict[species][species_T]["overlap"]
                ortholog_no_array[sp_idx, sp_idx_T] = intersection
    JI_DF = pd.DataFrame(JI_array,index=order, columns=order)
    gene_no_DF = pd.DataFrame(gene_no_array,index=order, columns=order)
    ortholog_no_DF = pd.DataFrame(ortholog_no_array,index=order, columns=order)
    return JI_DF, gene_no_DF, ortholog_no_DF
            
JI_DF, gene_no_DF, ortholog_no_DF = JI_dataframe(Ortholog_dict, species_total_genes_dict)


# %%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
#fig, ax = plt.subplots(figsize=(8, 6))
# Create masks
#plt.figure(figsize=(4, 4))
#JI
#sns.heatmap(JI_DF,  mask = np.array(JI_DF) == 0, cmap="Blues",  fmt=".4f",
#vmin = 0.2 , vmax =0.8 , cbar=True, cbar_kws={'label': '1-to-1 Orthologs (Jaccard Index)', "ticks": [0.2,0.5 ,0.8]}, annot = np.array(JI_DF))

#sns.heatmap(gene_no_DF,  mask = np.array(gene_no_DF) ==0, cmap= "Greys", fmt="d",
#vmin = 25_000 , vmax =35_000 , cbar=True, cbar_kws={'label': 'No. genes', "ticks": [25_000, 30_000 ,35_000 ]}, annot = np.array(gene_no_DF))

#sns.heatmap(ortholog_no_DF,  mask = np.array(ortholog_no_DF) ==0, cmap="Oranges", fmt="d",
# vmin = 19_000 , vmax =22_000 , cbar=True, cbar_kws={'label': 'No. 1-to-1 Orthologs', "ticks": [19_000, 20_500 ,22_000 ]}, annot = np.array(ortholog_no_DF))

#plt.savefig(os.path.join(OF_res_dir, "JI_Heatmap-CBAR.svg"))
# %%

#analyse conservation
information_dict = {"Arabidopsis_thaliana":{"dir" : "/mnt/md2/ken/CxNE_plants_data/taxid3702"},
"Arabidopsis_halleri":{"dir" : "/mnt/md2/ken/CxNE_plants_data/taxid81970"},
"Arabidopsis_lyrata":{"dir" : "/mnt/md2/ken/CxNE_plants_data/taxid59689"}}

# %%
import pickle
import scipy

def pickle_load(path):
    with open(path, "rb") as fbin:
        obj = pickle.load(fbin)
    return obj

# %%
#species_1 = "Arabidopsis_thaliana"
#species_2 = "Arabidopsis_halleri"
#adj_name = "adj_mat_zscore.pkl"

#adj_mat_1 =  pickle_load(os.path.join(information_dict[species_1]["dir"], adj_name))
#adj_mat_2 =  pickle_load(os.path.join(information_dict[species_2]["dir"], adj_name))
#gene_dict_1 = pickle_load(os.path.join(information_dict[species_1]["dir"], "gene_dict.pkl"))
#gene_dict_2 = pickle_load(os.path.join(information_dict[species_2]["dir"], "gene_dict.pkl"))

# %%
# subset expression matrix

#species_1_idx = np.array([gene_dict_1[key] for key in Ortholog_dict[species_1][species_2].keys() if key != "overlap"])
#species_2_idx =  np.array([gene_dict_2[Ortholog_dict[species_1][species_2][key]] for key in Ortholog_dict[species_1][species_2].keys()if key != "overlap"])
# %%
#SCC = scipy.stats.spearmanr(adj_mat_1[species_1_idx, :][:, species_1_idx] , adj_mat_2[species_2_idx, :][:, species_2_idx] , axis = None, alternative="greater").statistic
#PCC = scipy.stats.pearsonr(adj_mat_1[species_1_idx, :][:, species_1_idx] , adj_mat_2[species_2_idx, :][:, species_2_idx] , axis = None, alternative="greater").statistic
# %%

#scipy.stats.spearmanr([[1,2,3], [3,2,1]] , [[1,2,3], [3,2,1]] ,  axis = None, alternative="greater")

# %%

def PCC_SCC_dataframe(Ortholog_dict, information_dict, adj_name="adj_mat_zscore.pkl" , order=None):
    if order == None:
        order = list(Ortholog_dict.keys())
    PCC_array = np.zeros((len(order), len(order)), dtype="float32")
    SCC_array = np.zeros((len(order), len(order)), dtype="float32")
    for sp_idx_1 , species_1 in enumerate(order):
        adj_mat_1 =  pickle_load(os.path.join(information_dict[species_1]["dir"], adj_name))
        gene_dict_1 = pickle_load(os.path.join(information_dict[species_1]["dir"], "gene_dict.pkl"))
        #update gene_no_array
        for sp_idx_2 , species_2 in enumerate(order):
            if sp_idx_2 > sp_idx_1:
                adj_mat_2 =  pickle_load(os.path.join(information_dict[species_2]["dir"], adj_name))
                gene_dict_2 = pickle_load(os.path.join(information_dict[species_2]["dir"], "gene_dict.pkl"))
                species_1_idx = np.array([gene_dict_1[key] for key in Ortholog_dict[species_1][species_2].keys() if key != "overlap"])
                species_2_idx =  np.array([gene_dict_2[Ortholog_dict[species_1][species_2][key]] for key in Ortholog_dict[species_1][species_2].keys()if key != "overlap"])
                adj_mat_1_subset = adj_mat_1[species_1_idx, :][:, species_1_idx]
                adj_mat_2_subset = adj_mat_2[species_2_idx, :][:, species_2_idx]
                print(f"calc PCC b/w {species_1} and {species_2}...")
                PCC = scipy.stats.pearsonr(adj_mat_1_subset , adj_mat_2_subset , axis = None, alternative="greater").statistic
                print(f"calc SCC b/w {species_1} and {species_2}...")
                SCC = scipy.stats.spearmanr(adj_mat_1_subset , adj_mat_2_subset , axis = None, alternative="greater").statistic
                PCC_array[sp_idx_1, sp_idx_2] = PCC
                SCC_array[sp_idx_2, sp_idx_1] = SCC
    PCC_DF = pd.DataFrame(PCC_array,index=order, columns=order)
    SCC_DF = pd.DataFrame(SCC_array,index=order, columns=order)
    return PCC_DF, SCC_DF
            


# %%
PCC_DF , SCC_DF = PCC_SCC_dataframe(Ortholog_dict, information_dict, adj_name="adj_mat_zscore_PCC_k1_AVG.pkl" , order=None)

PCC_DF.to_csv(os.path.join(OF_res_dir, "PCC_DF_control.csv"))
SCC_DF.to_csv(os.path.join(OF_res_dir, "SCC_DF_control.csv"))

# %%
#load df
PCC_DF = pd.read_csv(os.path.join(OF_res_dir, "PCC_DF.csv")).set_index("Unnamed: 0")
SCC_DF = pd.read_csv(os.path.join(OF_res_dir, "SCC_DF.csv")).set_index("Unnamed: 0")
# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(4, 4))
sns.heatmap(SCC_DF,  mask = np.array(SCC_DF) == 0, cmap="BrBG",  fmt=".4f",
vmin =-1 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression Correlation (SCC)', "ticks": [-1, 0 ,1]}, annot = np.array(SCC_DF))

sns.heatmap(PCC_DF,  mask = np.array(PCC_DF) ==0, cmap="RdBu", fmt=".4f",
vmin = -1 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression Correlation (PCC)', "ticks": [-1, 0 ,1]}, annot = np.array(PCC_DF))

plt.ylabel("")
plt.savefig(os.path.join(OF_res_dir, "TEA_Co-expression_Correlation_Heatmap.svg"))
# %%
#Jaccard index above zscore 1
def determine_JI(adj_mat_1_subset, adj_mat_2_subset, threshold=1):
    adj_mat_1_subset_bool = adj_mat_1_subset >= threshold
    adj_mat_2_subset_bool = adj_mat_2_subset >= threshold
    intersection = int((adj_mat_1_subset_bool & adj_mat_2_subset_bool).astype(int).sum())
    union = int(adj_mat_1_subset_bool.astype(int).sum() + adj_mat_2_subset_bool.astype(int).sum()) - intersection
    JI = intersection/ union
    return JI

def JI_COEXP_dataframe(Ortholog_dict, information_dict, adj_name="adj_mat_zscore.pkl" , order=None):
    if order == None:
        order = list(Ortholog_dict.keys())
    JI_array_1 = np.zeros((len(order), len(order)), dtype="float32")
    JI_array_2 = np.zeros((len(order), len(order)), dtype="float32")
    for sp_idx_1 , species_1 in enumerate(order):
        adj_mat_1 =  pickle_load(os.path.join(information_dict[species_1]["dir"], adj_name))
        gene_dict_1 = pickle_load(os.path.join(information_dict[species_1]["dir"], "gene_dict.pkl"))
        #update gene_no_array
        for sp_idx_2 , species_2 in enumerate(order):
            if sp_idx_2 > sp_idx_1:
                adj_mat_2 =  pickle_load(os.path.join(information_dict[species_2]["dir"], adj_name))
                gene_dict_2 = pickle_load(os.path.join(information_dict[species_2]["dir"], "gene_dict.pkl"))
                species_1_idx = np.array([gene_dict_1[key] for key in Ortholog_dict[species_1][species_2].keys() if key != "overlap"])
                species_2_idx =  np.array([gene_dict_2[Ortholog_dict[species_1][species_2][key]] for key in Ortholog_dict[species_1][species_2].keys()if key != "overlap"])
                adj_mat_1_subset = adj_mat_1[species_1_idx, :][:, species_1_idx]
                adj_mat_2_subset = adj_mat_2[species_2_idx, :][:, species_2_idx]
                print(f"calc JI_1 b/w {species_1} and {species_2}...")
                JI_1 = determine_JI(adj_mat_1_subset,adj_mat_2_subset, threshold = 1)
                print(f"calc JI_2 b/w {species_1} and {species_2}...")
                JI_2 = determine_JI(adj_mat_1_subset,adj_mat_2_subset, threshold = 0.5)
                JI_array_1[sp_idx_1, sp_idx_2] = JI_1
                JI_array_2[sp_idx_2, sp_idx_1] = JI_2
    JI_1_DF = pd.DataFrame(JI_array_1,index=order, columns=order)
    JI_2_DF = pd.DataFrame(JI_array_2,index=order, columns=order)
    return JI_1_DF, JI_2_DF



# %%
JI_1_DF , JI_05_DF = JI_COEXP_dataframe(Ortholog_dict, information_dict, adj_name="adj_mat_zscore_PCC_k1_AVG.pkl" , order=None)

JI_1_DF.to_csv(os.path.join(OF_res_dir, "JI_1_DF_control.csv"))
JI_05_DF.to_csv(os.path.join(OF_res_dir, "JI_05_DF_control.csv"))

# %%
JI_1_DF = pd.read_csv(os.path.join(OF_res_dir, "JI_1_DF.csv")).set_index("Unnamed: 0")
JI_05_DF= pd.read_csv(os.path.join(OF_res_dir, "JI_05_DF.csv")).set_index("Unnamed: 0")
# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(4, 4))

sns.heatmap(JI_1_DF,  mask = np.array(JI_1_DF) == 0, cmap="Purples",  fmt=".4f",
vmin = 0 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression z-score >= 1 (Jaccard Index)', "ticks": [0, 0.5 ,1]}, annot = np.array(JI_1_DF))

sns.heatmap(JI_05_DF,  mask = np.array(JI_05_DF) ==0, cmap="Greens", fmt=".4f",
vmin = 0 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression z-score >= 0.5 (Jaccard Index)', "ticks": [0, 0.5 ,1]}, annot = np.array(JI_05_DF))

plt.ylabel("")
plt.savefig(os.path.join(OF_res_dir, "TEA_Co-expression_JI_Heatmap.svg"))
# %%
