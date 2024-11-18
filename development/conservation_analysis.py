# %%
import os
import sys

import pandas as pd
import numpy as np
import pickle
# %%
def pickle_load(path):
    with open(path, "rb") as fbin:
        obj = pickle.load(fbin)
    return obj

def pickle_write(obj, path):
    with open(path, "wb") as fbin:
        pickle.dump(obj, fbin)

#def edit_geneid(geneid):
#    return geneid.split(".")[0].upper()

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
                    source = line.split("\t")[1]
                    target = line.split("\t")[-1]
                    Ortholog_dict[sp][sp_target][source] = target
                    overlap += 1
            Ortholog_dict[sp][sp_target]["overlap"] = overlap
    return Ortholog_dict

def mine_orthologs(orthologues_dir,information_dict):
    Paralog_set = set()
    Ortholog_1v1_set = set()
    Ortholog_others_set = set()
    for sp in species:
        #loadtid
        tid2gid_dict_sp = pickle_load(os.path.join(information_dict[sp]["dir"], "Tid2Gid_dict.pkl"))
        sp_dir = os.path.join(orthologues_dir, f"Orthologues_{sp}")
        for file in os.listdir(sp_dir):
            sp_target = file.split("__v__")[-1].replace(".tsv", "")
            tid2gid_dict_sp_target = pickle_load(os.path.join(information_dict[sp_target]["dir"], "Tid2Gid_dict.pkl"))
            with open(os.path.join(sp_dir, file), "r") as fin:
                contents = fin.read().split("\n")
            for line in contents[1:]:
                if line != "":
                    sources = line.split("\t")[1].split(", ")
                    targets = line.split("\t")[-1].split(", ")
                    
                    for s_idx, source in enumerate(sources[:-1]):
                        for source_t in  sources[s_idx+1:]:
                            try:
                                Paralog_set.add("--".join(sorted([tid2gid_dict_sp[source],
                                                                tid2gid_dict_sp[source_t]])))
                            except:
                                pass
                    
                    for t_idx, target in enumerate(targets[:-1]):
                        for target_t in  targets[t_idx+1:]:
                            try:
                                Paralog_set.add("--".join(sorted([tid2gid_dict_sp_target[target],
                                                                tid2gid_dict_sp_target[target_t]])))
                            except:
                                pass

                    if len(sources) ==1 and len(targets)==1:
                        try:
                            Ortholog_1v1_set.add("--".join(sorted([tid2gid_dict_sp[sources[0]], 
                                                                tid2gid_dict_sp_target[targets[0]]])))
                        except:
                            pass
                    else:
                        for source in sources:
                            for target in targets:
                                try:
                                    Ortholog_others_set.add("--".join(sorted([tid2gid_dict_sp_target[target],
                                                                            tid2gid_dict_sp[source]])))
                                except:
                                    pass

    return Paralog_set, Ortholog_1v1_set, Ortholog_others_set

# %%

#extract orthofinder
OF_res_dir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/Orthofinder/ATTED_input_fasta/OF_results/Results_Oct08"
orthologues_dir = os.path.join(OF_res_dir , "Orthologues")
species = [folder.replace("Orthologues_", "") for folder in os.listdir(orthologues_dir) if "." not in folder]


# %%
OF_order = ["taxid59689",
         "taxid81970",
         "taxid3702",
         "taxid3711",
         "taxid29760",
         "taxid3694",
         "taxid4081",
         "taxid3880",
         "taxid39947",
         "taxid4577"]
information_dict = {spe: {"dir" : os.path.join("/mnt/md2/ken/CxNE_plants_data/species_data/", spe)} for spe in OF_order}

Paralog_set, Ortholog_1v1_set, Ortholog_others_set = mine_orthologs(orthologues_dir,information_dict)
# %%
print("Paralog_set:",len(Paralog_set))
print("Ortholog_1v1_set:",len(Ortholog_1v1_set))
print("Ortholog_others_set:",len(Ortholog_others_set))


# %%
#write relationships
paralog_list = list(Paralog_set)
#paralog first
paralog_path = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/angiosperm_human_yeast/IS_PARALOG_relationships.csv"

with open(paralog_path, "w") as fout:
    for edge in paralog_list:
        source , target = edge.split("--")
        line = f"{source},{target},IS_PARALOG\n"
        fout.write(line)

#Ortholog_1v1_set, Ortholog_others_set
Ortholog_1v1_list = list(Ortholog_1v1_set)
Ortholog_others_list = list(Ortholog_others_set)
#paralog first
ortholog_path = "/mnt/md2/ken/CxNE_plants_data/species_data/neo4j_input/angiosperm_human_yeast/IS_ORTHOLOG_relationships.csv"

with open(ortholog_path, "w") as fout:
    for edge in Ortholog_1v1_list:
        source , target = edge.split("--")
        line = f"{source},{target},true,IS_ORTHOLOG\n"
        fout.write(line)
    
    for edge in Ortholog_others_list:
        source , target = edge.split("--")
        line = f"{source},{target},false,IS_ORTHOLOG\n"
        fout.write(line)

# %%
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
# %%
OF_order = ["taxid59689",
         "taxid81970",
         "taxid3702",
         "taxid3711",
         "taxid29760",
         "taxid3694",
         "taxid4081",
         "taxid3880",
         "taxid39947",
         "taxid4577"]
# %%
JI_DF, gene_no_DF, ortholog_no_DF = JI_dataframe(Ortholog_dict, species_total_genes_dict, order=OF_order)
# %%
JI_DF.to_csv(os.path.join(OF_res_dir, "JI_DF.csv"))
gene_no_DF.to_csv(os.path.join(OF_res_dir, "gene_no_DF.csv"))
ortholog_no_DF.to_csv(os.path.join(OF_res_dir, "ortholog_no_DF.csv"))
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
if False:

    #fig, ax = plt.subplots(figsize=(8, 6))
    # Create masks
    plt.figure(figsize=(8, 8))
    #JI
    sns.heatmap(JI_DF,  mask = np.array(JI_DF) == 0, cmap="Blues",  fmt=".4f",
    vmin = 0.0 , vmax =0.6 , cbar=True, cbar_kws={'label': '1-to-1 Orthologs (Jaccard Index)', "ticks": [0,0.3 ,0.6]}, annot = np.array(JI_DF))

    sns.heatmap(gene_no_DF,  mask = np.array(gene_no_DF) ==0, cmap= "Greys", fmt="d",
    vmin = 0 , vmax =50_000 , cbar=True, cbar_kws={'label': 'No. genes', "ticks": [0, 25_000 ,50_000 ]}, annot = np.array(gene_no_DF))

    sns.heatmap(ortholog_no_DF,  mask = np.array(ortholog_no_DF) ==0, cmap="Oranges", fmt="d",
    vmin = 0 , vmax =30_000 , cbar=True, cbar_kws={'label': 'No. 1-to-1 Orthologs', "ticks": [0, 15_000 ,30_000 ]}, annot = np.array(ortholog_no_DF))

    plt.savefig(os.path.join(OF_res_dir, "JI_Heatmap_Cbar.svg"))
# %%

#analyse conservation
if False:
    information_dict = {"taxid3702":{"dir" : "/mnt/md2/ken/CxNE_plants_data/taxid3702"},
    "taxid81970":{"dir" : "/mnt/md2/ken/CxNE_plants_data/taxid81970"},
    "taxid59689":{"dir" : "/mnt/md2/ken/CxNE_plants_data/taxid59689"}}

# %%
information_dict = {spe: {"dir" : os.path.join("/mnt/md2/ken/CxNE_plants_data/species_data/", spe)} for spe in OF_order}

# %%
import pickle
import scipy

def pickle_load(path):
    with open(path, "rb") as fbin:
        obj = pickle.load(fbin)
    return obj

def pickle_write(obj, path):
    with open(path, "wb") as fbin:
        pickle.dump(obj, fbin)


# %%
#species_1 = "taxid3702"
#species_2 = "taxid59689"
#adj_name = "adj_mat_zscore_PCC_k1_AVG.pkl"

#adj_mat_1 =  pickle_load(os.path.join(information_dict[species_1]["dir"], adj_name))

#adj_mat_2 =  pickle_load(os.path.join(information_dict[species_2]["dir"], adj_name))
#gene_dict_1 = pickle_load(os.path.join(information_dict[species_1]["dir"], "gene_dict.pkl"))
#gene_dict_2 = pickle_load(os.path.join(information_dict[species_2]["dir"], "gene_dict.pkl"))

# %%
#adj_mat_1 = adj_mat_1* -1
#adj_mat_2 = adj_mat_2* -1

# %%
#pickle_write(adj_mat_1, os.path.join(information_dict[species_1]["dir"], adj_name))

#pickle_write(adj_mat_2, os.path.join(information_dict[species_2]["dir"], adj_name))

# %%
#species_1 = "taxid3694"
#gene_dict_1 = pickle_load(os.path.join(information_dict[species_1]["dir"], "gene_dict.pkl"))
#gene_dict_2 = pickle_load(os.path.join(information_dict[species_2]["dir"], "gene_dict.pkl"))
#tid2gid_dict_1 = pickle_load(os.path.join(information_dict[species_1]["dir"], "Tid2Gid_dict.pkl"))

#print(tid2gid_dict_1)
#print(Ortholog_dict[species_1][species_2])
#tid2gid_dict_2 = pickle_load(os.path.join(information_dict[species_2]["dir"], "Tid2Gid_dict.pkl"))

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

def PCC_SCC_dataframe(Ortholog_dict, information_dict, adj_name="adj_mat_zscore.pkl" , order=None, PCC_DF=None, SCC_DF = None):
    if order == None:
        order = list(Ortholog_dict.keys())
    if PCC_DF is None:
        PCC_array = np.zeros((len(order), len(order)), dtype="float32")
        PCC_array[:] = np.nan
    else:
        PCC_array = np.array(PCC_DF)
    if SCC_DF is None:
        SCC_array = np.zeros((len(order), len(order)), dtype="float32")
        SCC_array[:] = np.nan
    else:
        SCC_array = np.array(SCC_DF)
    for sp_idx_1 , species_1 in enumerate(order):
        try:
            adj_mat_1 =  pickle_load(os.path.join(information_dict[species_1]["dir"], adj_name))
            gene_dict_1 = pickle_load(os.path.join(information_dict[species_1]["dir"], "gene_dict.pkl"))
            tid2gid_dict_1 = pickle_load(os.path.join(information_dict[species_1]["dir"], "Tid2Gid_dict.pkl"))
            #update gene_no_array
            for sp_idx_2 , species_2 in enumerate(order):
                if sp_idx_2 > sp_idx_1:
                    adj_mat_2 =  pickle_load(os.path.join(information_dict[species_2]["dir"], adj_name))
                    gene_dict_2 = pickle_load(os.path.join(information_dict[species_2]["dir"], "gene_dict.pkl"))
                    tid2gid_dict_2 = pickle_load(os.path.join(information_dict[species_2]["dir"], "Tid2Gid_dict.pkl"))
                    species_1_idx = np.array([gene_dict_1[tid2gid_dict_1[key]] for key in Ortholog_dict[species_1][species_2].keys() if key != "overlap"])
                    species_2_idx =  np.array([gene_dict_2[tid2gid_dict_2[Ortholog_dict[species_1][species_2][key]]] for key in Ortholog_dict[species_1][species_2].keys()if key != "overlap"])
                    adj_mat_1_subset = adj_mat_1[species_1_idx, :][:, species_1_idx]
                    adj_mat_2_subset = adj_mat_2[species_2_idx, :][:, species_2_idx]
                    if not PCC_array[sp_idx_1, sp_idx_2] > -500:
                        print(f"calc PCC b/w {species_1} and {species_2}...")
                        PCC = scipy.stats.pearsonr(adj_mat_1_subset , adj_mat_2_subset , axis = None, alternative="greater").statistic
                        PCC_array[sp_idx_1, sp_idx_2] = PCC
                        print(species_1, species_2, "PCC", PCC )
                    if not SCC_array[sp_idx_2, sp_idx_1] > -500:
                        print(f"calc SCC b/w {species_1} and {species_2}...")
                        SCC = scipy.stats.spearmanr(adj_mat_1_subset , adj_mat_2_subset , axis = None, alternative="greater").statistic
                        SCC_array[sp_idx_2, sp_idx_1] = SCC
                        print(species_1, species_2, "SCC", SCC )
        except:
            pass

    PCC_DF = pd.DataFrame(PCC_array,index=order, columns=order)
    SCC_DF = pd.DataFrame(SCC_array,index=order, columns=order)
    return PCC_DF, SCC_DF
            


# %%
if False:
    print("adj_mat_zscore_PCC_k1_AVG, PCC, SCC...")
    PCC_DF , SCC_DF = PCC_SCC_dataframe(Ortholog_dict, information_dict, adj_name="adj_mat_zscore_PCC_k1_AVG.pkl" , order=OF_order)

    PCC_DF.to_csv(os.path.join(OF_res_dir, "PCC_DF_control.csv"))
    SCC_DF.to_csv(os.path.join(OF_res_dir, "SCC_DF_control.csv"))

# %%
#load df
if False:
    PCC_DF = pd.read_csv(os.path.join(OF_res_dir, "PCC_DF.csv")).set_index("Unnamed: 0")
    SCC_DF = pd.read_csv(os.path.join(OF_res_dir, "SCC_DF.csv")).set_index("Unnamed: 0")
    PCC_DF_control = pd.read_csv(os.path.join(OF_res_dir, "PCC_DF_control.csv")).set_index("Unnamed: 0")
    SCC_DF_control = pd.read_csv(os.path.join(OF_res_dir, "SCC_DF_control.csv")).set_index("Unnamed: 0")
# %%
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 8))
    sns.heatmap(SCC_DF,  mask = np.array(SCC_DF) == 0, cmap="BrBG",  fmt=".3f",
    vmin =-1 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression Correlation (SCC)', "ticks": [-1, 0 ,1]}, annot = np.array(SCC_DF))

    sns.heatmap(PCC_DF,  mask = np.array(PCC_DF) ==0, cmap="RdBu", fmt=".3f",
    vmin = -1 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression Correlation (PCC)', "ticks": [-1, 0 ,1]}, annot = np.array(PCC_DF))

    plt.ylabel("")
    #plt.savefig(os.path.join(OF_res_dir, "TEA_Co-expression_Correlation_Heatmap.svg"))
# %%    
    plt.figure(figsize=(8, 8))
    sns.heatmap(SCC_DF_control,  mask = np.array(SCC_DF_control) == 0, cmap="BrBG",  fmt=".3f",
    vmin =-1 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression Correlation (SCC)', "ticks": [-1, 0 ,1]}, annot = np.array(SCC_DF_control))

    sns.heatmap(PCC_DF_control,  mask = np.array(PCC_DF_control) ==0, cmap="RdBu", fmt=".3f",
    vmin = -1 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression Correlation (PCC)', "ticks": [-1, 0 ,1]}, annot = np.array(PCC_DF_control))

    plt.ylabel("")
    plt.savefig(os.path.join(OF_res_dir, "Control_Co-expression_Correlation_Heatmap.svg"))
# %%
#Jaccard index above zscore 1
def determine_JI(adj_mat_1_subset, adj_mat_2_subset, threshold=1):
    adj_mat_1_subset_bool = adj_mat_1_subset >= threshold
    adj_mat_2_subset_bool = adj_mat_2_subset >= threshold
    intersection = int((adj_mat_1_subset_bool & adj_mat_2_subset_bool).astype(int).sum())
    union = int(adj_mat_1_subset_bool.astype(int).sum() + adj_mat_2_subset_bool.astype(int).sum()) - intersection
    JI = intersection/ union
    return JI

def JI_COEXP_dataframe(Ortholog_dict, information_dict, adj_name="adj_mat_zscore.pkl" , order=None, JI_1_DF=None, JI_2_DF = None):
    if order == None:
        order = list(Ortholog_dict.keys())
    if JI_1_DF is None:
        JI_array_1 = np.zeros((len(order), len(order)), dtype="float32")
        JI_array_1[:] = np.nan
    else:
         JI_array_1 = np.array(JI_1_DF)
    if JI_2_DF is None:
        JI_array_2 = np.zeros((len(order), len(order)), dtype="float32")
        JI_array_2[:] = np.nan
    else:
        JI_array_2 = np.array(JI_1_DF)
    for sp_idx_1 , species_1 in enumerate(order):
        try:
            adj_mat_1 =  pickle_load(os.path.join(information_dict[species_1]["dir"], adj_name))
            gene_dict_1 = pickle_load(os.path.join(information_dict[species_1]["dir"], "gene_dict.pkl"))
            tid2gid_dict_1 = pickle_load(os.path.join(information_dict[species_1]["dir"], "Tid2Gid_dict.pkl"))
            #update gene_no_array
            for sp_idx_2 , species_2 in enumerate(order):
                if sp_idx_2 > sp_idx_1:
                    adj_mat_2 =  pickle_load(os.path.join(information_dict[species_2]["dir"], adj_name))
                    gene_dict_2 = pickle_load(os.path.join(information_dict[species_2]["dir"], "gene_dict.pkl"))
                    tid2gid_dict_2 = pickle_load(os.path.join(information_dict[species_2]["dir"], "Tid2Gid_dict.pkl"))
                    species_1_idx = np.array([gene_dict_1[tid2gid_dict_1[key]] for key in Ortholog_dict[species_1][species_2].keys() if key != "overlap"])
                    species_2_idx =  np.array([gene_dict_2[tid2gid_dict_2[Ortholog_dict[species_1][species_2][key]]] for key in Ortholog_dict[species_1][species_2].keys()if key != "overlap"])
                    adj_mat_1_subset = adj_mat_1[species_1_idx, :][:, species_1_idx]
                    adj_mat_2_subset = adj_mat_2[species_2_idx, :][:, species_2_idx]
                    if not JI_array_1[sp_idx_1, sp_idx_2] > -500:
                        print(f"calc JI_1 b/w {species_1} and {species_2}...")
                        JI_1 = determine_JI(adj_mat_1_subset,adj_mat_2_subset, threshold = 1)
                        JI_array_1[sp_idx_1, sp_idx_2] = JI_1
                        print(species_1, species_2, "JI_1", JI_1 )
                    if not JI_array_2[sp_idx_2, sp_idx_1] > -500:
                        print(f"calc JI_2 b/w {species_1} and {species_2}...")
                        JI_2 = determine_JI(adj_mat_1_subset,adj_mat_2_subset, threshold = 0.5)
                        JI_array_2[sp_idx_2, sp_idx_1] = JI_2
                        print(species_1, species_2, "JI_05", JI_2 )
        except:
            pass
    JI_1_DF = pd.DataFrame(JI_array_1,index=order, columns=order)
    JI_2_DF = pd.DataFrame(JI_array_2,index=order, columns=order)
    return JI_1_DF, JI_2_DF



# %%
if False:
    print("adj_mat_zscore_PCC_k1_AVG, JI...")
    JI_1_DF , JI_05_DF = JI_COEXP_dataframe(Ortholog_dict, information_dict, adj_name="adj_mat_zscore_PCC_k1_AVG.pkl" , order=OF_order)

    JI_1_DF.to_csv(os.path.join(OF_res_dir, "JI_1_DF_control.csv"))
    JI_05_DF.to_csv(os.path.join(OF_res_dir, "JI_05_DF_control.csv"))

if True:
    PCC_DF = pd.read_csv(os.path.join(OF_res_dir, "PCC_DF_new.csv")).set_index("Unnamed: 0")
    SCC_DF = pd.read_csv(os.path.join(OF_res_dir, "SCC_DF_new.csv")).set_index("Unnamed: 0")
    print("adj_mat_zscore, PCC, SCC...")
    PCC_DF , SCC_DF = PCC_SCC_dataframe(Ortholog_dict, information_dict, adj_name="adj_mat_zscore.pkl" , order=OF_order,
                                        PCC_DF= PCC_DF,
                                        SCC_DF= SCC_DF )

    PCC_DF.to_csv(os.path.join(OF_res_dir, "PCC_DF_newer.csv"))
    SCC_DF.to_csv(os.path.join(OF_res_dir, "SCC_DF_newer.csv"))

    JI_1_DF = pd.read_csv(os.path.join(OF_res_dir, "JI_1_DF_new.csv")).set_index("Unnamed: 0")
    JI_05_DF = pd.read_csv(os.path.join(OF_res_dir, "JI_05_DF_new.csv")).set_index("Unnamed: 0")
    print("adj_mat_zscore, JI...")
    JI_1_DF , JI_05_DF = JI_COEXP_dataframe(Ortholog_dict, information_dict, adj_name="adj_mat_zscore.pkl" , order=OF_order,
                                             JI_1_DF=JI_1_DF, JI_2_DF = JI_05_DF)

    JI_1_DF.to_csv(os.path.join(OF_res_dir, "JI_1_DF_newer.csv"))
    JI_05_DF.to_csv(os.path.join(OF_res_dir, "JI_05_DF_newer.csv"))


sys.exit()
# %%
JI_1_DF = pd.read_csv(os.path.join(OF_res_dir, "JI_1_DF_newer.csv")).set_index("Unnamed: 0")
JI_05_DF= pd.read_csv(os.path.join(OF_res_dir, "JI_05_DF_newer.csv")).set_index("Unnamed: 0")
JI_1_DF_control = pd.read_csv(os.path.join(OF_res_dir, "JI_1_DF_control.csv")).set_index("Unnamed: 0")
JI_05_DF_control= pd.read_csv(os.path.join(OF_res_dir, "JI_05_DF_control.csv")).set_index("Unnamed: 0")
PCC_DF = pd.read_csv(os.path.join(OF_res_dir, "PCC_DF_newer.csv")).set_index("Unnamed: 0")
SCC_DF= pd.read_csv(os.path.join(OF_res_dir, "SCC_DF_newer.csv")).set_index("Unnamed: 0")
PCC_DF_control = pd.read_csv(os.path.join(OF_res_dir, "PCC_DF_control.csv")).set_index("Unnamed: 0")
SCC_DF_control= pd.read_csv(os.path.join(OF_res_dir, "SCC_DF_control.csv")).set_index("Unnamed: 0")
# %%
import matplotlib.pyplot as plt
import seaborn as sns
# %%
plt.figure(figsize=(8, 8))

sns.heatmap(JI_1_DF,  mask = np.array(JI_1_DF) == 0, cmap="Purples",  fmt=".3f",
vmin = 0 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression z-score >= 1 (Jaccard Index)', "ticks": [0, 0.5 ,1]}, annot = np.array(JI_1_DF))

sns.heatmap(JI_05_DF,  mask = np.array(JI_05_DF) ==0, cmap="Greens", fmt=".3f",
vmin = 0 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression z-score >= 0.5 (Jaccard Index)', "ticks": [0, 0.5 ,1]}, annot = np.array(JI_05_DF))

plt.ylabel("")
#plt.savefig(os.path.join(OF_res_dir, "TEA_Co-expression_JI_Heatmap.svg"))
# %%
plt.figure(figsize=(8, 8))

sns.heatmap(JI_1_DF_control,  mask = np.array(JI_1_DF_control) == 0, cmap="Purples",  fmt=".3f",
vmin = 0 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression z-score >= 1 (Jaccard Index)', "ticks": [0, 0.5 ,1]}, annot = np.array(JI_1_DF_control))

sns.heatmap(JI_05_DF_control,  mask = np.array(JI_05_DF_control) ==0, cmap="Greens", fmt=".3f",
vmin = 0 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression z-score >= 0.5 (Jaccard Index)', "ticks": [0, 0.5 ,1]}, annot = np.array(JI_05_DF_control))

plt.ylabel("")
plt.savefig(os.path.join(OF_res_dir, "Control_Co-expression_JI_Heatmap.svg"))

# %%

#somemore ploting
plt.figure(figsize=(8, 8))
sns.heatmap(JI_1_DF_control,  mask = np.array(JI_1_DF_control) ==0, cmap="Purples", fmt=".3f",
vmin = 0 , vmax =1 , cbar=True, cbar_kws={'label': 'Ortholog Co-expression z-score >= 1 (Jaccard Index)', "ticks": [0, 0.5 ,1]}, annot = np.array(JI_1_DF_control))

sns.heatmap(SCC_DF_control,  mask = np.array(SCC_DF_control) == 0, cmap="Oranges",  fmt=".3f",
vmin =0 , vmax =1 , cbar=True, cbar_kws={'label': 'Ortholog Co-expression Correlation (SCC)', "ticks": [0 ,0.5,1]}, annot = np.array(SCC_DF_control))
plt.ylabel("")
plt.savefig(os.path.join(OF_res_dir, "Control_hybrid_heatmap_newest_cbar.svg"))

# %%
#somemore ploting
plt.figure(figsize=(8, 8))
sns.heatmap(JI_1_DF,  mask = np.array(JI_1_DF) ==0, cmap="Purples", fmt=".3f",
vmin = 0 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression z-score >= 1 (Jaccard Index)', "ticks": [0, 0.5 ,1]}, annot = np.array(JI_1_DF))

sns.heatmap(SCC_DF,  mask = np.array(SCC_DF) == 0, cmap="Oranges",  fmt=".3f",
vmin =0 , vmax =1 , cbar=False, cbar_kws={'label': 'Ortholog Co-expression Correlation (SCC)', "ticks": [0 ,0.5,1]}, annot = np.array(SCC_DF))
plt.ylabel("")
plt.savefig(os.path.join(OF_res_dir, "TEA_hybrid_heatmap_newest.svg"))
# %%
from scipy import stats, integrate
import numpy as np
from sklearn import metrics
import pandas as pd
import math

def tie_break(scores):
    scores = [score - (offset/100000000000) for offset, score in enumerate(scores)]
    return scores

def calc_AUC_ROC(pos_score, neg_score, return_thresholds =False):
    scores = pos_score + neg_score
    labels = [1 for i in pos_score] + [0 for i in neg_score]    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    AUC_ROC = metrics.auc(fpr, tpr)
    if return_thresholds:
        return fpr, tpr, thresholds, AUC_ROC
    else:
        return AUC_ROC

def calc_AUC_PRC(pos_score, neg_score, return_thresholds =False):
    scores = tie_break(pos_score) + tie_break(neg_score)
    labels = [1 for i in pos_score] + [0 for i in neg_score]    
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=1)
    AUC_PRC= metrics.auc(recall, precision)
    #AUC_PRC = np.abs(integrate.trapz(y=precision, x=recall))
    if return_thresholds:
        return precision, recall, thresholds, AUC_PRC
    else:
        return AUC_PRC

import math

def extract_edge_score(edges, score_dict, score_type):
    """extract scores for a particcular score type"""
    scores = []
    for edge in edges:
        try:
            scores.append(score_dict[edge][score_type])
        except:
            scores.append(math.nan)
            print(f"WARNING: {edge} not found in score_dict!")
    return scores

def nan2value(scores, n=-1):
    """convert nan values to something else"""
    new_scores = [-1 if np.isnan(i) else i for i in scores]
    return new_scores

def calc_quartiles(performance_scores):
    return_array = list(stats.scoreatpercentile(performance_scores, [25,50,75], interpolation_method = "lower"))
    return return_array

def evaluate(positive_edges_cor_dict, negative_edges_cor_dict, positive_edges , negative_edges, score_types):
    """Main function."""
    performance_dict = {}
    for score_type in score_types:
        performance_dict[score_type] = {"AUC_ROC":{},"AUC_PRC":{}, "AVG":{}}
        positive_scores = nan2value(extract_edge_score( positive_edges, positive_edges_cor_dict, score_type))
        for ds , negative_edges_cor_dict_values  in negative_edges_cor_dict.items():
            negative_scores = nan2value(extract_edge_score(negative_edges[ds], negative_edges_cor_dict_values, score_type))
            performance_dict[score_type]["AUC_ROC"][ds] = float(calc_AUC_ROC(positive_scores, negative_scores, return_thresholds =False))
            performance_dict[score_type]["AUC_PRC"][ds] = float(calc_AUC_PRC(positive_scores, negative_scores, return_thresholds =False))
            performance_dict[score_type]["AVG"][ds] = float(np.mean([performance_dict[score_type]["AUC_ROC"][ds], performance_dict[score_type]["AUC_PRC"][ds]]))
        performance_dict[score_type]["Quartiles"] = {}
        performance_dict[score_type]["Quartiles"]["AUC_ROC"] = calc_quartiles(list(performance_dict[score_type]["AUC_ROC"].values()))
        performance_dict[score_type]["Quartiles"]["AUC_PRC"] = calc_quartiles(list(performance_dict[score_type]["AUC_PRC"].values()))
        performance_dict[score_type]["Quartiles"]["AVG"] = calc_quartiles(list(performance_dict[score_type]["AVG"].values()))
        #calc full thresholds
        performance_dict[score_type]["Thresholds"]={}
        ds_AUROC = list(performance_dict[score_type]["AUC_ROC"].values()).index(performance_dict[score_type]["Quartiles"]["AUC_ROC"][1])
        negative_scores = nan2value(extract_edge_score(negative_edges[ds_AUROC], negative_edges_cor_dict[ds_AUROC], score_type))
        performance_dict[score_type]["Thresholds"]["AUC_ROC"] = calc_AUC_ROC(positive_scores, negative_scores, return_thresholds =True)

        ds_AUPRC = list(performance_dict[score_type]["AUC_PRC"].values()).index(performance_dict[score_type]["Quartiles"]["AUC_PRC"][1])
        negative_scores = nan2value(extract_edge_score(negative_edges[ds_AUPRC], negative_edges_cor_dict[ds_AUPRC], score_type))
        performance_dict[score_type]["Thresholds"]["AUC_PRC"] = calc_AUC_PRC(positive_scores, negative_scores, return_thresholds =True)

    return performance_dict
# %%
import pickle
outdir = "/mnt/md2/ken/CxNE_plants_data/species_data/taxid81970/"
gene_dict_path = os.path.join(outdir,"gene_dict.pkl" )
with open(gene_dict_path, "rb") as fin:
    gene_dict = pickle.load(fin)

adj_mat_zscore_path = os.path.join(outdir, "adj_mat_zscore.pkl")
with open(adj_mat_zscore_path, "rb") as fin:
    adj_mat_zscore = pickle.load( fin)
# %%
adj_mat_zscore = adj_mat_zscore.astype("float32")
# %%
# load negative and postiive edges
label_dir = "/mnt/md0/ken/correlation_networks/Plant-GCN_data/PMN_b1_1/taxid81970/Label_edges"
positive_met_edges_path  = os.path.join(label_dir,"positive_met_edges.pkl")
negative_met_edges_path = os.path.join(label_dir,"negative_met_edges.pkl")

with open(positive_met_edges_path, "rb") as fin:
    positive_met_edges = pickle.load( fin)

with open(negative_met_edges_path, "rb") as fin:
    negative_met_edges = pickle.load( fin)
# %%
positive_edges_cor_dict = {}
negative_edges_cor_dict = {}

for edge in positive_met_edges:
    source , target = edge.split("--")
    source_idx, target_idx = gene_dict[source], gene_dict[target]
    score = adj_mat_zscore[source_idx, target_idx]
    positive_edges_cor_dict[edge] = {"RAW": float(score)}

for ds, edges in negative_met_edges.items():
    negative_edges_cor_dict[ds] ={}
    for edge in edges:
        source , target = edge.split("--")
        source_idx, target_idx = gene_dict[source], gene_dict[target]
        score = adj_mat_zscore[source_idx, target_idx]
        negative_edges_cor_dict[ds][edge] = {"RAW": float(score)}
# %%
performance_dict_TEA = evaluate(positive_edges_cor_dict, negative_edges_cor_dict, positive_met_edges , negative_met_edges, ["RAW"])
print("TEA-performance:")
print("AUC_ROC:", float(performance_dict_TEA["RAW"]["Quartiles"]["AUC_ROC"][1]))
print("AUC_PRC:", float(performance_dict_TEA["RAW"]["Quartiles"]["AUC_PRC"][1]))
performance_dict_TEA_path = os.path.join(outdir, "performance_dict_TEA.pkl")
with open(performance_dict_TEA_path, "wb") as fbout:
    pickle.dump(performance_dict_TEA, fbout)
# %%
adj_mat_zscore_path = os.path.join(outdir, "adj_mat_zscore_PCC_k1_AVG.pkl")
with open(adj_mat_zscore_path, "rb") as fin:
    adj_mat_zscore = pickle.load( fin)
# %%
adj_mat_zscore = adj_mat_zscore.astype("float32")

positive_edges_cor_dict = {}
negative_edges_cor_dict = {}

for edge in positive_met_edges:
    source , target = edge.split("--")
    source_idx, target_idx = gene_dict[source], gene_dict[target]
    score = adj_mat_zscore[source_idx, target_idx]
    positive_edges_cor_dict[edge] = {"RAW": float(score)}

for ds, edges in negative_met_edges.items():
    negative_edges_cor_dict[ds] ={}
    for edge in edges:
        source , target = edge.split("--")
        source_idx, target_idx = gene_dict[source], gene_dict[target]
        score = adj_mat_zscore[source_idx, target_idx]
        negative_edges_cor_dict[ds][edge] = {"RAW": float(score)}
# %%
performance_dict_PCC = evaluate(positive_edges_cor_dict, negative_edges_cor_dict, positive_met_edges , negative_met_edges, ["RAW"])
print("PCC-performance:")
print("AUC_ROC:", float(performance_dict_PCC["RAW"]["Quartiles"]["AUC_ROC"][1]))
print("AUC_PRC:", float(performance_dict_PCC["RAW"]["Quartiles"]["AUC_PRC"][1]))
performance_dict_PCC_path = os.path.join(outdir, "performance_dict_PCC.pkl")
with open(performance_dict_PCC_path, "wb") as fbout:
    pickle.dump(performance_dict_PCC, fbout)
# %%

#plot box and whiskers
import os
OF_order = ["taxid59689",
         "taxid81970",
         "taxid3702",
         "taxid3711",
         "taxid29760",
         "taxid3694",
         "taxid4081",
         "taxid3880",
         "taxid39947",
         "taxid4577"]
workdir = "/mnt/md2/ken/CxNE_plants_data/species_data/"
organisms = {"taxid59689": {"Name": "A. lyrata"},
             "taxid81970":{"Name": "A. halleri"},
             "taxid3702": {"Name": "A. thaliana"},
             "taxid3711": {"Name":"B. rapa"},
             "taxid29760":{"Name": "V. vinifera"},
             "taxid3694":{"Name": "P. trichocarpa"},
             "taxid4081":{"Name": "S. lycopersicum"},
             "taxid3880":{"Name": "M. truncatula"},
             "taxid39947":{"Name": "O. sativa"},
             "taxid4577":{"Name": "Z. mays" }}

import seaborn as sns

#hue will be the different networks
# classes will be the different organisms
#do for AUROC and AUROC

import pandas as pd
# %%

data = pd.DataFrame()
#init cols

AUROC_col =[]
AUPRC_col= []
network_col = []
organism_col = []

import pickle


for taxid, info in organisms.items():
    sci_name = info["Name"]
    PCC_performance_dict_path = os.path.join(workdir,taxid, "performance_dict_PCC.pkl")
    with open(PCC_performance_dict_path, "rb") as fin:
        PCC_performance_dict = pickle.load(fin)
    
    for i in range(100):
        if taxid != "taxid3702":
            AUROC = PCC_performance_dict["RAW"]["AUC_ROC"][i]
            AUPRC = PCC_performance_dict["RAW"]["AUC_PRC"][i]
        else:
            AUROC = PCC_performance_dict["Avg"]["Met"]["AUC_ROC"][i]
            AUPRC = PCC_performance_dict["Avg"]["Met"]["AUC_PRC"][i]

        AUROC_col.append(AUROC)
        AUPRC_col.append(AUPRC)
        network_col.append("PCC")
        organism_col.append(sci_name)

    
    #TEA
    TEA_performance_dict_path = os.path.join(workdir,taxid,"performance_dict_TEA.pkl" )
    with open(TEA_performance_dict_path, "rb") as fin:
        TEA_performance_dict = pickle.load(fin)
    for i in range(100):
        if taxid != "taxid3702":
            AUROC = TEA_performance_dict["RAW"]["AUC_ROC"][i]
            AUPRC = TEA_performance_dict["RAW"]["AUC_PRC"][i]
        else:
            AUROC = TEA_performance_dict["Met"]["AUC_ROC"][i]
            AUPRC = TEA_performance_dict["Met"]["AUC_PRC"][i]
            
        AUROC_col.append(AUROC)
        AUPRC_col.append(AUPRC)
        network_col.append("TEA-GCN")
        organism_col.append(sci_name)
        
data["Species"] =  organism_col
data["GCN"] = network_col
data["AUROC"] = AUROC_col
data["AUPRC"] = AUPRC_col

# %%
import matplotlib.pyplot as plt

scaling_factor = 0.3
plt.figure(figsize=(15* scaling_factor,20* scaling_factor))
plt.rcParams['axes.linewidth'] = 0.5
sns.boxplot(data = data , x ="Species" ,y="AUROC", hue= "GCN",
            palette={"TEA-GCN": sns.color_palette()[2] , "PCC": sns.color_palette()[0]}, 
            saturation =0.75 ,linewidth=0.5, fliersize = 0, width=0.5, dodge = False)
plt.legend().remove()
plt.xticks( [])
plt.yticks( fontsize=8)
#plt.yticks( fontsize=8)
plt.xlabel( "")
plt.ylabel("")
plt.ylim(0.44,0.86)
plt.savefig(os.path.join(workdir, "ROC_conservation.svg"))
plt.show()
plt.clf()


# %%
import matplotlib.pyplot as plt

scaling_factor = 0.3
plt.figure(figsize=(15* scaling_factor,20* scaling_factor))
plt.rcParams['axes.linewidth'] = 0.5
sns.boxplot(data = data , x ="Species" ,y="AUPRC", hue= "GCN",
            palette={"TEA-GCN": sns.color_palette()[2] , "PCC": sns.color_palette()[0]}, 
            saturation =0.75 ,linewidth=0.5, fliersize = 0, width=0.5, dodge = False)
plt.legend().remove()
plt.xticks( [])
plt.yticks( fontsize=8)
#plt.yticks( fontsize=8)
plt.xlabel( "")
plt.ylabel("")
plt.ylim(0.44,0.86)
plt.savefig(os.path.join(workdir, "PRC_conservation.svg"))
plt.show()
plt.clf()

# %%

scaling_factor = 2.2
plt.figure(figsize=(2* scaling_factor,1* scaling_factor))
plt.rcParams['axes.linewidth'] = 0.5
sns.violinplot(data=data, x="Species", y="AUPRC", hue= "GCN",
               palette={"TEA-GCN": "#7DE290" , "PCC": "#90B1E0"}, 
               linewidth=0.25, width=0.5, dodge = False)

plt.legend().remove()
plt.yticks([0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85] ,fontsize=6)
plt.xlabel( "")
plt.ylabel("")
plt.ylim(0.44,0.86)
plt.savefig(os.path.join(workdir, "PRC_conservation_violin.svg"))
plt.show()
plt.clf()

# %%
scaling_factor = 2.2
plt.figure(figsize=(2* scaling_factor,1* scaling_factor))
plt.rcParams['axes.linewidth'] = 0.5
sns.violinplot(data=data, x="Species", y="AUROC", hue= "GCN",
               palette={"TEA-GCN": "#7DE290" , "PCC": "#90B1E0"}, 
               linewidth=0.25, width=0.5, dodge = False)

plt.legend().remove()
plt.yticks([0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85] ,fontsize=6)
plt.xlabel( "")
plt.ylabel("")
plt.ylim(0.44,0.86)
plt.savefig(os.path.join(workdir, "ROC_conservation_violin.svg"))
plt.show()
plt.clf()
# %%
