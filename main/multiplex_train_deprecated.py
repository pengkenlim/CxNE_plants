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
import gdown
import torch
import multiprocessing
from multiprocessing import Queue
import queue
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import ClusterData, ClusterLoader
import numpy as np
#from torch.cuda.amp import autocast
from torch.amp import autocast, GradScaler
import joblib

from utils import others, models, loss_func
from joblib import Parallel, delayed
import joblib

def infer(rearranged_out, loader):
    with torch.no_grad():
        #model.eval()
        with autocast(device_type='cuda'if train_param.mode == "GPU" else "cpu", dtype=torch.float16 if train_param.precision == "HALF" else torch.float32):
            for i in range(train_param.inference_replicates):
                for batch_idx , batch in enumerate(loader):
                    batch = batch.to(GPU_device)
                    out = model(batch.x,  
                                batch.edge_index, 
                                batch.edge_weight)
                    out= out.to(CPU_device)
                    rearranged_out[batch.y.to(CPU_device)] += out
                torch.cuda.empty_cache()
        infer_out = rearranged_out / train_param.inference_replicates
    return infer_out

def prepare_data_deprec(spe):
    partition_save_dir = os.path.join(train_param.data_dir,spe,"Cluster-GCN")
    if not os.path.exists(partition_save_dir):
        os.makedirs(partition_save_dir)
    input_graph_path = os.path.join(train_param.data_dir, spe, train_param.input_graph_name)
    input_graph = others.pickle_load(input_graph_path)
    coexp_adj_mat_path = os.path.join(train_param.data_dir, spe, train_param.coexp_adj_mat_name)
    coexp_adj_mat = others.pickle_load(coexp_adj_mat_path)
    cluster_data = ClusterData(input_graph, num_parts  = train_param.clusterGCN_num_parts, save_dir= partition_save_dir, keep_inter_cluster_edges =True)
    loader= ClusterLoader(cluster_data, num_workers = train_param.num_workers, batch_size = train_param.clusterGCN_parts_perbatch, shuffle=True)
    true_out = torch.zeros(input_graph.x.size()[0], train_param.decode_kwargs["out_channels"])
    return coexp_adj_mat, loader, true_out, input_graph


def load_cached_file(file_path):
    return joblib.load(file_path)

# Load files concurrently using Joblib's Parallel
#results = Parallel(n_jobs=len(file_paths), backend='threading')(delayed(load_cached_file)(file_path) for file_path in file_paths)

def prepare_data(spe):
    cache_dir = os.path.join(train_param.data_dir,spe,"cache")
    loader_path = os.path.join(cache_dir, "loader.joblib")
    coexp_adj_mat_path = os.path.join(cache_dir, "loader.joblib")
    node_split_masks_path = os.path.join(cache_dir, "node_split_masks.joblib")
    file_paths = [loader_path, coexp_adj_mat_path,  node_split_masks_path]
    results = Parallel(n_jobs=len(file_paths), backend='threading')(delayed(load_cached_file)(file_path) for file_path in file_paths)
    return results

results = prepare_data("taxid3702")
x = np.zeros((1000,1000))
joblib.dump(x , "/home/ken/x.joblib")
x= x.astype("float16")
joblib.dump(x , "/home/ken/x16.joblib")

input_graph.x

loader = load_cached_file(loader_path)


def cache_and_cluster_data(spe):
    partition_save_dir = os.path.join(train_param.data_dir,spe,"Cluster-GCN")
    if not os.path.exists(partition_save_dir):
        os.makedirs(partition_save_dir)
    cache_dir = os.path.join(train_param.data_dir,spe,"cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(spe," Loading input_graph...")
    input_graph_path = os.path.join(train_param.data_dir, spe, train_param.input_graph_name)
    input_graph = others.pickle_load(input_graph_path)
    print(spe," Loading coexp_adj_mat...")
    coexp_adj_mat_path = os.path.join(train_param.data_dir, spe, train_param.coexp_adj_mat_name)
    coexp_adj_mat = others.pickle_load(coexp_adj_mat_path)
    print(spe," Clustering...")
    cluster_data = ClusterData(input_graph, num_parts  = train_param.clusterGCN_num_parts, save_dir= partition_save_dir, keep_inter_cluster_edges =True)
    loader= ClusterLoader(cluster_data, num_workers = train_param.num_workers, batch_size = train_param.clusterGCN_parts_perbatch, shuffle=True)
    node_split_masks = {"y": input_graph.y,
                       "train_mask" : input_graph.train_mask,
                       "val_mask" : input_graph.val_mask,
                       "test_mask" : input_graph.test_mask}
    #dump
    print(spe," dumping loader...")
    loader_path = os.path.join(cache_dir, "loader.joblib")
    joblib.dump(loader, loader_path)
    print(spe," dumping coexp_adj_mat...")
    coexp_adj_mat_path = os.path.join(cache_dir, "coexp_adj_mat.joblib")
    joblib.dump(coexp_adj_mat, coexp_adj_mat_path)
    print(spe," dumping node_split_masks...")
    node_split_masks_path = os.path.join(cache_dir, "node_split_masks.joblib")
    joblib.dump(node_split_masks, node_split_masks_path)




def train_on_minibatches(batches, spe, true_out, epoch, coexp_adj_mat):
    model.train()
    for mb_idx, subgraph in enumerate(batches):
        subgraph = subgraph.to(GPU_device)
        learn_rate = scheduler.get_last_lr()[-1]
        optimizer.zero_grad()
        out = model(subgraph.x,  subgraph.edge_index, subgraph.edge_weight)
        #relating to training loss....
        train_out = out[subgraph.train_mask]
        if train_param.precision == "HALF":
            with autocast(device_type='cuda'if train_param.mode == "GPU" else "cpu", dtype=torch.float16):
                RMSE = loss_func.RMSE_dotprod_vs_coexp(train_out, subgraph.y[subgraph.train_mask], coexp_adj_mat, GPU_device)
                scaler.scale(RMSE).backward()
                scaler.step(optimizer)
                scaler.update()
        elif train_param.precision == "FULL":
                RMSE = loss_func.RMSE_dotprod_vs_coexp(train_out, subgraph.y[subgraph.train_mask], coexp_adj_mat, GPU_device)
                RMSE.backward()
                optimizer.step()
        print(f"Mini-Batch Performance | epoch {epoch} |species {spe} | mini-batch {mb_idx} | tr_loss: {RMSE:6f} lr: {learn_rate}")
        with open(log_path, "a") as logout:
            logout.write(f"{epoch}\t{spe}\t{mb_idx}\tMini_Batch_Performance\t{RMSE:6f}\t-\t-\t{learn_rate}\n")
        true_out[subgraph.y.to(CPU_device)] = out.detach().to(CPU_device)
    #clear cache
    torch.cuda.empty_cache()
    return true_out, learn_rate

def main(number_of_epochs):
    for epoch in range(number_of_epochs):
        model.train()
        epoch_train_RMSE, epoch_val_RMSE, epoch_test_RMSE = [],[],[]
        for spe in species:
            #data preprocess
            coexp_adj_mat, loader, true_out, input_graph = prepare_data(spe)
            batches = [batch for batch in loader]
            #train model
            true_out, learn_rate = train_on_minibatches(batches, spe, true_out, epoch, coexp_adj_mat)
            #evaluate output
            spe_train_RMSE, spe_val_RMSE, spe_test_RMSE = loss_func.evaluate_output(input_graph.y, true_out, 
                                                                        input_graph.train_mask, 
                                                                        input_graph.val_mask, 
                                                                        input_graph.test_mask, 
                                                                        coexp_adj_mat)
            epoch_train_RMSE.append(spe_train_RMSE)
            epoch_val_RMSE.append(spe_val_RMSE)
            epoch_test_RMSE.append(spe_test_RMSE)
            print(f"Species-specific epoch Performance| Epoch {epoch} | species {spe} | tr_loss: {spe_train_RMSE:6f}, val_loss: {spe_val_RMSE:6f}, tst_loss: {spe_test_RMSE:6f}, lr: {learn_rate}")
            with open(log_path, "a") as logout:
                logout.write(f"{epoch}\t{spe}\t-\tSpecies-specific epoch Performance\t{spe_train_RMSE:6f}\t{spe_val_RMSE:6f}\t{spe_test_RMSE:6f}\t{learn_rate}\n")
        epoch_train_RMSE = float(np.array(epoch_train_RMSE).mean())
        epoch_val_RMSE = float(np.array(epoch_val_RMSE).mean())
        epoch_test_RMSE = float(np.array(epoch_test_RMSE).mean())
        print(f"Epoch Performance| Epoch {epoch} | tr_loss: {epoch_train_RMSE:6f}, val_loss: {epoch_val_RMSE:6f}, tst_loss: {epoch_test_RMSE:6f}, lr: {learn_rate}")
        with open(log_path, "a") as logout:
            logout.write(f"{epoch}\t-\t-\tEpoch Performance\t{epoch_train_RMSE:6f}\t{epoch_val_RMSE:6f}\t{epoch_test_RMSE:6f}\t{learn_rate}\n")

        if epoch > train_param.inference_start and epoch % train_param.inference_interval == 0:
            print(f"epoch: {epoch}| Proceeding with Inference for evaluation")
            infer_epoch_train_RMSE, infer_epoch_val_RMSE, infer_epoch_test_RMSE = [],[],[]
            for spe in species:
                coexp_adj_mat, loader, infer_out, input_graph = prepare_data(spe)
                infer_out = infer(infer_out, loader)
                infer_spe_train_RMSE, infer_spe_val_RMSE, infer_spe_test_RMSE = loss_func.evaluate_output(input_graph.y.to(CPU_device), infer_out, 
                                                                        input_graph.train_mask.to(CPU_device), 
                                                                        input_graph.val_mask.to(CPU_device), 
                                                                        input_graph.test_mask.to(CPU_device), 
                                                                        coexp_adj_mat)
                with open(log_path, "a") as logout:
                    logout.write(f"{epoch}\t{spe}\t-\tSpecies-specific epoch Performance (inference mode)\t{infer_spe_train_RMSE:6f}\t{infer_spe_val_RMSE:6f}\t{infer_spe_test_RMSE:6f}\t{learn_rate}\n")
                print(f"Species-specific epoch Performance (inference mode) | Epoch {epoch} | species {spe}| tr_loss: {infer_spe_train_RMSE:6f}, val_loss: {infer_spe_val_RMSE:6f}, tst_loss: {infer_spe_test_RMSE:6f}, lr: {learn_rate}")
                infer_epoch_train_RMSE.append(infer_spe_train_RMSE)
                infer_epoch_val_RMSE.append(infer_spe_val_RMSE)
                infer_epoch_test_RMSE.append(infer_spe_test_RMSE)
                if train_param.save_inference_embeddings:
                    infer_dump_sub_dir = os.path.join(infer_dump_dir, spe)
                    if not os.path.exists(infer_dump_sub_dir):
                        os.makedirs(infer_dump_sub_dir)
                    embeddings_path = os.path.join(infer_dump_sub_dir, f"Epoch{epoch}_emb.pkl")
                    with open(embeddings_path, "wb") as fbout:
                        pickle.dump(infer_out, fbout)
            infer_epoch_train_RMSE = float(np.array(infer_epoch_train_RMSE).mean())
            infer_epoch_val_RMSE = float(np.array(infer_epoch_val_RMSE).mean())
            infer_epoch_test_RMSE = float(np.array(infer_epoch_test_RMSE).mean())
            print(f"Epoch Performance (inference mode) | Epoch {epoch} | tr_loss: {infer_epoch_train_RMSE:6f}, val_loss: {infer_epoch_val_RMSE:6f}, tst_loss: {infer_epoch_test_RMSE:6f}, lr: {learn_rate}")
            with open(log_path, "a") as logout:
                logout.write(f"{epoch}\t-\t-\tEpoch Performance (inference mode)\t{infer_epoch_train_RMSE:6f}\t{infer_epoch_val_RMSE:6f}\t{infer_epoch_test_RMSE:6f}\t{learn_rate}\n")
            if infer_epoch_train_RMSE < train_param.checkpoint_threshold_loss:
                print(f"Inference training loss for this epoch {infer_epoch_train_RMSE:4f} is lower than threshold of {train_param.checkpoint_threshold_loss}. Saving model...")
                model_state_path = os.path.join(model_dump_dir, f"Epoch{epoch}_model_state.pth")
                torch.save(model.state_dict(), model_state_path)


if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="CxNE_plants/main/multiple_train.py. Train a Graph Neural Network-based Model to learn Gene co-expression embeddings (CxNE) ACROSS multiple species.")
    
    parser.add_argument("-p", "--param_path",  type=str ,required = True,
                        help= "File path to parameters needed to run multiple_train.py.")
    
    #load params
    args=parser.parse_args()
    param_path = args.param_path
    train_param = others.parse_parameters(param_path)
    #train_param = others.parse_parameters("/home/ken/CxNE_plants/param_templates/multiplex_train_param.py")
    
    #downloading data
    if train_param.data_download_link is not None:
        print(f" Downloading from {train_param.data_download_link} to {train_param.data_dir}...")
        gdown.download_folder(url=train_param.data_download_link, output = train_param.data_dir )

    #create_outdir
    if not os.path.exists(train_param.output_dir):
        os.makedirs(train_param.output_dir)
    #copy training parameters
    shutil.copy(param_path, os.path.join(train_param.output_dir, "multiplex_train_param.py"))

    #species
    if train_param.species == "All" or None:
        species = os.listdir(train_param.data_dir)
    else:
        species  = train_param.species

    #clearing cache
    torch.cuda.empty_cache()

    #initializing model
    print("Innitializing weights of model...\n")
    model = models.CxNE(**train_param.CxNE_kwargs)
    #model = CxNE(**train_param.CxNE_kwargs) #remove
    print("Architecture of model is as follows:\n")
    print(model)
    print("done\n")

    #Initializing optimizer and defining device
    print("Initializing optimizer and defining device...")
    if train_param.mode == "CPU":
        GPU_device = torch.device("cpu")
    elif train_param.mode == "GPU":
        GPU_device= torch.device('cuda')
    
    CPU_device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), **train_param.optimizer_kwargs)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', **train_param.scheduler_kwargs)
    print("done\n")
    

    #creating log file
    log_path = os.path.join(train_param.output_dir,"training_logs.txt")
    with open(log_path, "a") as logout:
         logout.write("Epoch\tSpecies\tBatch\tMode\ttrain_loss\tval_loss\ttest_loss\tlearn_rate\n")
    
    #create subdir to dump model states
    model_dump_dir = os.path.join(train_param.output_dir,"model_states_dump")
    if not os.path.exists(model_dump_dir):
        os.makedirs(model_dump_dir)
    
    #create subdir to dump inference embeddings
    infer_dump_dir = os.path.join(train_param.output_dir,"infered_emmbeddings")
    if not os.path.exists(infer_dump_dir):
        os.makedirs(infer_dump_dir)

    
    model = model.to(GPU_device)
    #scaler = torch.cuda.amp.GradScaler()
    if train_param.mode == "CPU":
        scaler = GradScaler('cpu')
    elif train_param.mode == "GPU":
        scaler = GradScaler("cuda")
    
    #prepare and cluster data
    for spe in species:
        cache_and_cluster_data(spe)

    #main function
    #main(train_param.num_epoch)






             








    

    