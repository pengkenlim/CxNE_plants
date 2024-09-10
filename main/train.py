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

from utils import others, models

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

    


    

    