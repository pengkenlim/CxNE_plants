# %%
import torch
from torch_geometric.loader import RandomNodeLoader

import pickle
import time
# %%
#load data
path = "/mnt/md2/ken/CxNE_plants_data/species_data/taxid3702/adj_mat_zscore_20percent_ESM3B_data.pkl"
with open(path, "rb") as fbin:
    data =pickle.load(fbin) 
# %%

train_loader = RandomNodeLoader(data, num_parts=100, shuffle=True,
                                num_workers=64)
# %%
start = time.time()
subgraph = next(iter(train_loader))
print( (time.time() - start)/ 60, "minutes")
# %%
for idx, subgraph in enumerate(train_loader):
    start = time.time()
    subgraph = next(iter(train_loader))
    print("idx:",idx, (time.time() - start)/ 60, "minutes")
    
# %%
