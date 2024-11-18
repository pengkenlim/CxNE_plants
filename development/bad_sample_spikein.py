# %%
import os
import sys

import pandas as pd

# %%
path_to_qc = "/mnt/md0/wallace_data/ATTED_b2/qc-out/taxid3702_qc_info.tsv"

with open(path_to_qc, "r") as fin:
    contents = fin.read().split("\n")
qc_dict = {line.split("\t")[0]: [float(line.split("\t")[1]), int(line.split("\t")[2])] for line in contents if line != ""}
# %%
len(qc_dict)
# %%
import numpy as np
qc_array = np.array(list(qc_dict.values()))
qc_labels = list(qc_dict.keys())
# %%
#load
path_to_exp = "/mnt/md0/wallace_data/ATTED_b2/qc-matrices/taxid3702_expmat.tsv"

exp_mat = pd.read_csv(path_to_exp, sep = "\t")
# %%
#find failed samples
failed_samples = {key: item for key, item in qc_dict.items() if item[0] < 20}

failed_samples_nps1k = {key: item for key, item in failed_samples.items() if item[1] >= 1_000}


# %%
passed_samples = {key: item for key, item in qc_dict.items() if item[0] >= 20}


# %%
#sample without replacement
bad_2500 = [str(i) for i in list(np.random.choice(list(failed_samples_nps1k.keys()), size = 2500, replace = False))]

bad_2000 = [str(i) for i in list(np.random.choice(bad_2500, size = 2000, replace = False))]
            
bad_1500 = [str(i) for i in list(np.random.choice(bad_2000, size = 1500, replace = False))]

bad_1000 = [str(i) for i in list(np.random.choice(bad_1500, size = 1000, replace = False))]

bad_500 = [str(i) for i in list(np.random.choice(bad_1000, size = 500, replace = False))]   

# %%
#load exp_mat1.5k
exp_mat_1_5K = pd.read_csv( "/mnt/md2/ken/BS_spikein/taxid3702_5k_1.5k/QC_expression_data/expression_matrix.tsv" ,sep = "\t").set_index("Unnamed: 0")

# %%
#load exp_mat2.5k
exp_mat_2_5K = pd.read_csv( "/mnt/md2/ken/BS_spikein/taxid3702_5k_2.5k/QC_expression_data/expression_matrix.tsv" ,sep = "\t").set_index("Unnamed: 0")


# %%
#i want to make 2k
diff = set(list(exp_mat_2_5K.columns)).difference(set(list(exp_mat_1_5K.columns)))
bad_500 = [str(i) for i in list(np.random.choice(list(diff), size = 500, replace = False))]  

# %%
exp_mat_sub = exp_mat.loc[:, bad_500]
exp_mat_bs = pd.concat([exp_mat_1_5K, exp_mat_sub], ignore_index=False, axis = 1)
exp_mat_bs.to_csv("/mnt/md2/ken/BS_spikein/taxid3702_5k_2k/QC_expression_data/expression_matrix.tsv", sep = "\t")
# %%
#exp_mat = exp_mat.T
exp_mat_good = pd.read_csv( "/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b2/taxid3702_5k/QC_expression_data/expression_matrix.tsv" ,sep = "\t").set_index("Unnamed: 0")
# %%
exp_mat_sub = exp_mat.loc[:, bad_2500]

exp_mat_bs = pd.concat([exp_mat_good, exp_mat_sub], ignore_index=False, axis = 1)

# %%
exp_mat_bs.to_csv("/mnt/md2/ken/BS_spikein/taxid3702_5k_2.5k/QC_expression_data/expression_matrix.tsv", sep = "\t")
# %%
#load eval

# %%
import os
import pickle

outputdir="/mnt/md2/ken/BS_spikein/taxid3702_5k_2k"
#outputdir="/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b2/taxid3702_5k"
network_name= "combine_all_15k_RAvg"
mode = "MR"

performance_dict_path  = os.path.join(outputdir, "Evaluate_full_network",network_name, "performance_dict.pkl")

with open(performance_dict_path, "rb") as fbin:
    performance_dict =  pickle.load(fbin)

# %%
print(outputdir, network_name, mode)
print(performance_dict[mode]["HM"]["Quartiles"]["AUC_ROC"][1])
print(performance_dict[mode]["HM"]["Quartiles"]["AUC_PRC"][1])

# %%
outputdir="/mnt/md0/ken/correlation_networks/Plant-GCN_data/ATTED_b2/taxid3702_5k"
network_name= "combine_all_18k_RAvg"
mode = "MR"




