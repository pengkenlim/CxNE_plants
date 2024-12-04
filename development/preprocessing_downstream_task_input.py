# %%
#step 1 generate labels for multiclassfication
import pandas as pd
import os
import pickle
import numpy as np
from collections import Counter
workdir = "/mnt/md2/ken/CxNE_plants_data/species_data/taxid3702"
path = os.path.join(workdir, "3702_spm_new.csv")
gene_dict_path = os.path.join(workdir, "gene_dict.pkl")

with open(gene_dict_path, "rb") as fbin:
    gene_dict = pickle.load(fbin)

spm_df = pd.read_csv(path).set_index("gene").dropna(axis=1)
# %%
def label_generator(spm_df, tolerance=0.2):
    classes =  list(spm_df.columns)
    classes2labels_dict = {clAss :idx+1 for idx, clAss in enumerate(classes)}
    classes2labels_dict["Not Tissue-specific"] = 0
    labels2classes_dict = {value: key for key, value in classes2labels_dict.items()}
    gene_spm_classes_dict = {}
    list_to_count = []
    for idx, row in spm_df.iterrows():
        try:
            gene_idx = gene_dict[idx]
            values = np.array(row, dtype="float32")
            top_2 = np.sort(np.partition(values, -2)[-2:])
            second , first = top_2[0], top_2[1]
            if (second / first) < (1-tolerance): # if there is a larger than tolerance % diff b/w first and second value
                label = np.where(values==first)[0][0] + 1 #shift down by one as non-spec is 0
            else:
                label = 0 # non-specific
            gene_spm_classes_dict[idx] = int(label)
            list_to_count.append(labels2classes_dict[int(label)])       
        except:
            pass
    label_counts = Counter(list_to_count)
    return labels2classes_dict, label_counts, gene_spm_classes_dict


# %%
labels2classes_dict, label_counts, gene_labels_dict = label_generator(spm_df, tolerance=0)
print(label_counts)
n = 0
for gene in gene_dict.keys():
    try:
        catch = gene_labels_dict[gene]
    except:
        gene_labels_dict[gene] = 0
        n+= 1

print(n)
label_dir = os.path.join(workdir,"labels" ,"multi_class_0ptolerance")
    
labels = torch.zeros((len(gene_labels_dict)), dtype=torch.long)
for gene , idx in gene_dict.items():
        labels[idx] = gene_labels_dict[gene]



# %%
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

labels2classes_dict_path = os.path.join(label_dir , "labels2classes_dict.pkl")

gene_labels_dict_path = os.path.join(label_dir , "gene_labels_dict.pkl")

with open( labels2classes_dict_path, "wb") as fbout:
    pickle.dump(labels2classes_dict, fbout)

with open( gene_labels_dict_path, "wb") as fbout:
    pickle.dump(gene_labels_dict, fbout)

with open(os.path.join(label_dir, "labels.pkl"), "wb") as fbout:
        pickle.dump(labels,fbout)

# %%
species = "taxid3702"
datasetprefix = "ESM3B_concat_RP11_E500"
ESMname = "ESM3B_node_features.pkl"
modelname = "Runpod_model_11"
CxNE_epoch = 500

speciesdir = f"/mnt/md2/ken/CxNE_plants_data/species_data/{species}/"
datasetdir = os.path.join(speciesdir, "datasets", datasetprefix)
modeldir = f"/mnt/md2/ken/CxNE_plants_data/multiplex_models/{modelname}"
CxNE_path = os.path.join(modeldir, "Embeddings", species, f"Epoch{CxNE_epoch}_emb.pkl")
ESM_path = os.path.join(speciesdir, ESMname)
# %%
#load
import torch
with open(CxNE_path, "rb") as fbin:
    CxNE = torch.tensor(pickle.load(fbin), dtype=torch.float32)

with open(ESM_path, "rb") as fbin:
    ESM =  torch.tensor(pickle.load(fbin), dtype=torch.float32)
# %%
concatenated_features_DS1 = torch.cat((CxNE, ESM), axis = 1)
shuffled_ESM = ESM[torch.randperm(ESM.size(0))]
shuffled_CxNE = CxNE[torch.randperm(CxNE.size(0))]
concatenated_features_DS2 = torch.cat((CxNE, shuffled_ESM), axis = 1)
concatenated_features_DS3 = torch.cat((shuffled_CxNE, ESM), axis = 1)
concatenated_features_DS4 = torch.cat((shuffled_CxNE, shuffled_ESM), axis = 1)
# %%
#save features
with open(os.path.join(datasetdir + f"_DS1.pkl"), "wb") as fbout:
    pickle.dump(concatenated_features_DS1 , fbout)

with open(os.path.join(datasetdir + f"_DS2.pkl"), "wb") as fbout:
    pickle.dump(concatenated_features_DS2 , fbout)

with open(os.path.join(datasetdir + f"_DS3.pkl"), "wb") as fbout:
    pickle.dump(concatenated_features_DS3 , fbout)

with open(os.path.join(datasetdir + f"_DS4.pkl"), "wb") as fbout:
    pickle.dump(concatenated_features_DS4 , fbout)
# %%

###to be incoporated
if False:
    #remove
    gene_dict_path = f"/mnt/md2/ken/CxNE_plants_data/species_data/{species}/gene_dict.pkl"    
    with open(gene_dict_path, "rb") as fbin:
        gene_dict = pickle.load(fbin)
    
    labels = torch.zeros((len(gene_labels_dict)), dtype=torch.int32)
    for gene , idx in gene_dict.items():
        labels[idx] = gene_labels_dict[gene]
    with open(os.path.join(labeldir, "labels.pkl"), "wb") as fbout:
        pickle.dump(labels,fbout)
    