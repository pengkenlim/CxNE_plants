# %%
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import seaborn as sns
# %%
np.random.seed(42)
num_nodes = 100
num_features = 5   # Number of neighbor features
embedding_dims = 10  # Number of embedding dimensions
# %%
# Simulated neighbor features (X)
neighbor_features = np.random.rand(num_nodes, num_features) 
# %%
# Simulated embeddings (Y)
embeddings = np.random.rand(num_nodes, embedding_dims)
# %%
# Step 2: PLSR Model
# -------------------
pls = PLSRegression(n_components=5)  # Choose the number of components
pls.fit(neighbor_features, embeddings)
# %%
coefficients = pls.coef_ 
# %%
# Step 3: Visualize Coefficients
# -------------------------------
# Create a heatmap of coefficients
feature_names = [f"Feature_{i+1}" for i in range(num_features)]
embedding_names = [f"Embedding_{j+1}" for j in range(embedding_dims)]

# Convert coefficients to DataFrame for better visualization
coef_df = pd.DataFrame(coefficients, index=embedding_names, columns=feature_names)
# %%
# Heatmap of coefficients
plt.figure(figsize=(12, 8))
sns.heatmap(coef_df, annot=True, cmap="coolwarm", center=0)
plt.title("PLSR Coefficients: Neighbor Features vs Embedding Dimensions")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Neighbor Features")
plt.show()
# %%
import pickle
path_to_embeddings = "/mnt/md2/ken/CxNE_plants_data/multiplex_models/Runpod_model_11/Embeddings/taxid3702/Epoch500_emb.pkl"
path_to_Adj = "/mnt/md2/ken/CxNE_plants_data/species_data/taxid3702/adj_mat_zscore.pkl"

with open(path_to_Adj, "rb") as fbin:
    Adj = pickle.load(fbin)

with open(path_to_embeddings, "rb") as fbin:
    emb = pickle.load(fbin)

Adj = Adj.astype("float32")
emb = emb.numpy()
# %%
pls = PLSRegression(n_components=96)
pls.fit(Adj, emb)
# %%
coefficients = pls.coef_ 
# %%
#save pls
import os
pls_dir = "/mnt/md2/ken/CxNE_plants_data/multiplex_models/Runpod_model_11/PLSR/taxid3702/Epoch500_emb/"
#os.makedirs(pls_dir)
pls_path = os.path.join(pls_dir, "pls.pkl")
# %%
with open(pls_path, "wb") as fbout:
    pickle.dump(pls, fbout)
# %%
coefficients_path = os.path.join(pls_dir, "coefficients.pkl")
with open(coefficients_path, "wb") as fbout:
    pickle.dump(coefficients, fbout)
# %%
#load genedict
gene_dict_path = "/mnt/md2/ken/CxNE_plants_data/species_data/taxid3702/gene_dict.pkl"
with open(gene_dict_path, "rb") as fbin:
    gene_dict = pickle.load(fbin)
# %%
import pandas as pd
gene_dict_flipped = {value:key for key, value in gene_dict.items()}
coef_df = pd.DataFrame(coefficients, columns = [gene_dict_flipped[i] for i in range(len(gene_dict_flipped)) ])
# %%
coef_df.to_csv(os.path.join(pls_dir, "coef_df.tsv"),sep = "\t")
# %%
# function definitions
import gseapy as gp
from gseapy import gseaplot
from gseapy import gseaplot2
import pandas as pd
import pickle
import os
def get_gene_sets(obo_path,GAF_path, category = "P", GeneID_col = 1):
  """returns gene_sets in the form of a dictionary.
  Use Category = "p", "M" or "C" to extract biological processes, molecular function, cellular component GOs respectively.
  GeneID_col = 1 or 2 to specify columns in the GAF file to use a GeneIDs
  format=  {"GO:1234567": ["Gene1", "Gene2", ...]...}"""
  GO_desc_dict = {}
  with open (obo_path, "r") as fin:
    contents = fin.read().split("[Term]\n")[1:]
    for chunk in contents:
      chunk_contents = chunk.split("\n")
      GO_desc_dict[chunk_contents[0].split("id: ")[-1].replace("\n", "")]= chunk_contents[1].split("name: ")[-1].replace("\n", "")

  with open(GAF_path, "r") as fin:
    annot_dict = {}
    for line_no , line in enumerate(fin):
        if line[0] != "!":
            line_contents= line.replace("\n", "").split("\t")
            if line_contents[8] == category:
                gene = line_contents[GeneID_col]
                GO_ID = line_contents[4]
                GO_ID_plus_desc = f"{GO_desc_dict[GO_ID]} ({GO_ID})"
                try:
                    annot_dict[GO_ID_plus_desc].append(gene)
                except:
                    annot_dict[GO_ID_plus_desc] = [gene]

  for GO_ID_plus_desc in annot_dict.keys():
      annot_dict[GO_ID_plus_desc] = set(annot_dict[GO_ID_plus_desc])
  return annot_dict
def get_signature(df, emb_idx):
  row = df.loc[emb_idx]
  signature= pd.DataFrame(row)
  return signature
# %%
obo_path = os.path.join(pls_dir, "go.obo")
GAF_path = os.path.join(pls_dir, "tair.gaf" )
annot_dict = get_gene_sets(obo_path, GAF_path, category = "P", GeneID_col = 1)
# %%
signature = get_signature(coef_df,4)

min_gene_set_size = 5
max_gene_set_size = 200
weight = 1
permutation_num = 1000
pre_res = gp.prerank(rnk=signature,  gene_sets = annot_dict,
                     threads=64, min_size = min_gene_set_size, max_size= max_gene_set_size,
                     permutation_num=permutation_num, verbose=True, weight = weight, ascending=False)

result = pre_res.res2d

results_filtered = result[result["FDR q-val"] <= 0.05]
results_filtered

# %%

# %%
