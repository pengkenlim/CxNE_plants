#setting sys.path for importing modules
import os
import sys

if __name__ == "__main__":
         abspath= __file__
         parent_module= "/".join(abspath.split("/")[:-2])
         sys.path.insert(0, parent_module)
         #print(parent_module) # remove

#sys.path.insert(0, "/home/ken/CxNE_plants") # remove

import argparse
import shutil
import pickle
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import GraphSAINTRandomWalkSampler
import numpy as np

from utils import others, models, loss_func

if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="CxNE_plants/main/train.py. Train a Graph Neural Network-based Model to learn Gene co-expression embeddings (CxNE).")
    
    parser.add_argument("-p", "--param_path",  type=str ,required = True,
                        help= "File path to parameters needed to run train.py.")
    
    #load params
    args=parser.parse_args()
    param_path = args.param_path
    train_param = others.parse_parameters(param_path)
    
    #create_outdir
    if not os.path.exists(train_param.output_dir):
        os.makedirs(train_param.output_dir)
    #copy training parameters
    shutil.copy(param_path, os.path.join(train_param.output_dir, "train_param.py"))

    #initializing model
    print("Innitializing weights of model...\n")
    model = models.CxNE(**train_param.CxNE_kwargs)
    print("Architecture of model is as follows:\n")
    print(model)
    print(model.parameters())
    print("done\n")

    #Initializing optimizer and defining device
    print("Initializing optimizer and defining device...")
    if train_param.mode == "CPU":
        device = torch.device("cpu")
    elif train_param.mode == "GPU":
         torch.device('cuda')
    optimizer = torch.optim.Adam(model.parameters(), **train_param.optimizer_kwargs)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', **train_param.scheduler_kwargs)
    print("done\n")
    
    #reading data
    print("loading data...")
    with open(train_param.input_graph_path, "rb") as finb:
        input_graph = pickle.load(finb)
    print(f"input_graph at {train_param.input_graph_path} loaded")

    with open(train_param.coexp_adj_mat, "rb") as finb:
        coexp_adj_mat = pickle.load(finb)
    print(f"coexp_adj_mat at {train_param.coexp_adj_mat} loaded")
    print("done\n")

    #creating log file
    log_path = os.path.join(train_param.output_dir,"training_logs.txt")
    with open("log_path", "a") as logout:
         logout.write("Epoch\tBatch\ttrain_loss\ttrain_loss_aggregated\tfull_batch_train_loss\tfull_batch_val_loss\n")
    
    
    
    #proceeding with training
    print("Creating minibatches...")
    save_dir = os.path.join(train_param.output_dir,"GraphSAINTRandomWalkSampler")
    if not os.path.exists(save_dir):
         os.makedirs(save_dir)
    loader = GraphSAINTRandomWalkSampler(input_graph,
                                     save_dir=save_dir,
                                     num_workers=train_param.num_workers,
                                     **train_param.datasampler_kwargs)

    epoch_train_performance = {}
    model.train()
    model.to(device)
    for epoch in range(train_param.num_epoch):
        total_summed_SE, total_num_contrasts = 0, 0
        for mb_idx, subgraph in enumerate(loader):
            subgraph.to(device)
            optimizer.zero_grad()
            out = model(subgraph.x,  subgraph.edge_index, subgraph.edge_weight)
            train_out = out[subgraph.train_mask]
            RMSE = loss_func.RMSE_dotprod_vs_coexp(train_out, subgraph.y[subgraph.train_mask], coexp_adj_mat)
            RMSE.backward()
            optimizer.step()
            scheduler.step(RMSE)
            num_contrasts = (((train_out.shape[0] **2) - train_out.shape[0])/2) + train_out.shape[0]
            RMSE = float(RMSE.detach())
            summed_SE = (RMSE**2)*num_contrasts
            total_num_contrasts += num_contrasts 
            total_summed_SE += summed_SE
            print(mb_idx, RMSE)
            with open("log_path", "a") as logout:
                logout.write(f"{epoch}\t{mb_idx}\t{RMSE:6f}\t-\t-\n")
        train_loss_aggregated = float(round(np.sqrt(total_summed_SE / total_num_contrasts), 6))
        print(f"epoch {epoch}, RMSE across batches: {train_loss_aggregated}")
        with open("log_path", "a") as logout:
            logout.write(f"{epoch}\t{mb_idx}\t-\t{train_loss_aggregated:6f}\t-\n")

    


    

    