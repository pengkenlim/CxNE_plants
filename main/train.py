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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import ClusterData, ClusterLoader
import numpy as np
from torch.cuda.amp import autocast

from utils import others, models, loss_func


@torch.no_grad()
def infer():
    with torch.no_grad():
        model.eval()
        rearranged_out = torch.zeros(input_graph.x.size()[0], train_param.decode_kwargs["out_channels"])
        for i in range(train_param.inference_replicates):
            for batch_idx , batch in enumerate(loader):
                out = model(batch.x.to(GPU_device),  
                            batch.edge_index.to(GPU_device), 
                            batch.edge_weight.to(GPU_device))
                rearranged_out[batch.y] += out
        infer_out = rearranged_out / train_param.inference_replicates
        return infer_out

if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="CxNE_plants/main/train.py. Train a Graph Neural Network-based Model to learn Gene co-expression embeddings (CxNE).")
    
    parser.add_argument("-p", "--param_path",  type=str ,required = True,
                        help= "File path to parameters needed to run train.py.")
    
    #load params
    args=parser.parse_args()
    param_path = args.param_path
    train_param = others.parse_parameters(param_path)
    #train_param = others.parse_parameters("/home/ken/CxNE_plants/train_param_skynet_test.py")
    
    #downloading data
    if train_param.input_graph_link is not None and not os.path.exists(train_param.input_graph_path):
        print(f"input graph not found. Downloading from {train_param.input_graph_link}...")
        gdown(train_param.input_graph_link, train_param.input_graph_path)

    if train_param.coexp_adj_mat_link is not None and not os.path.exists(train_param.coexp_adj_mat):
        print(f"Co-expression adjacency matrix not found. Downloading from {train_param.coexp_adj_mat_link}...")
        gdown(train_param.coexp_adj_mat_link, train_param.coexp_adj_mat)

    #create_outdir
    if not os.path.exists(train_param.output_dir):
        os.makedirs(train_param.output_dir)
    #copy training parameters
    shutil.copy(param_path, os.path.join(train_param.output_dir, "train_param.py"))

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
    with open(log_path, "a") as logout:
         logout.write("Epoch\tBatch\tMode\ttrain_loss\tval_loss\ttest_loss\tlearn_rate\n")
    
    #create subdir to dump model states
    model_dump_dir = os.path.join(train_param.output_dir,"model_states_dump")
    if not os.path.exists(model_dump_dir):
        os.makedirs(model_dump_dir)
    
    #create subdir to dump inference embeddings
    infer_dump_dir = os.path.join(train_param.output_dir,"infered_emmbeddings")
    if not os.path.exists(infer_dump_dir):
        os.makedirs(infer_dump_dir)


    #cluster-GCN
    save_dir = os.path.join(train_param.output_dir,"Cluster-GCN")
    if not os.path.exists(save_dir):
         os.makedirs(save_dir)
    cluster_data = ClusterData(input_graph, num_parts  = train_param.clusterGCN_num_parts, save_dir= save_dir, keep_inter_cluster_edges =True)
    loader= ClusterLoader(cluster_data, num_workers = train_param.num_workers, batch_size = train_param.clusterGCN_parts_perbatch, shuffle=True)

    print("\nGenerating subgraph batches to infer embeddings during evaluation...")




    model.train()
    model.to(GPU_device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(train_param.num_epoch):
        total_summed_SE_training, total_num_contrasts_training = 0, 0
        total_summed_SE_val, total_num_contrasts_val = 0, 0
        total_summed_SE_test, total_num_contrasts_test = 0, 0
        for mb_idx, subgraph in enumerate(loader):
            subgraph.to(GPU_device)
            learn_rate = scheduler.get_last_lr()[-1]
            optimizer.zero_grad()
            out = model(subgraph.x,  subgraph.edge_index, subgraph.edge_weight)
            
            #relating to training loss....
            train_out = out[subgraph.train_mask]
            if train_param.precision == "HALF":
                with autocast():
                    RMSE = loss_func.RMSE_dotprod_vs_coexp(train_out, subgraph.y[subgraph.train_mask], coexp_adj_mat)
                scaler.scale(RMSE).backward()
                scaler.step(optimizer)
                scaler.update()
            elif train_param.precision == "FULL":
                RMSE = loss_func.RMSE_dotprod_vs_coexp(train_out, subgraph.y[subgraph.train_mask], coexp_adj_mat)
                RMSE.backward()
                optimizer.step()
            num_contrasts_training = (((train_out.shape[0] **2) - train_out.shape[0])/2) + train_out.shape[0]
            RMSE = float(RMSE.detach())
            summed_SE = (RMSE**2)*num_contrasts_training
            total_num_contrasts_training += num_contrasts_training
            total_summed_SE_training += summed_SE
            
            #print(mb_idx, RMSE)

            #detach output after training
            out = out.detach()
            out.to(CPU_device)
            train_out = out[subgraph.train_mask]

            #relating to validation loss....
            
            val_out = out[subgraph.val_mask]
            val_out.to(CPU_device)
            RMSE_val, num_contrasts_val = loss_func.RMSE_dotprod_vs_coexp_testval(val_out, subgraph.y[subgraph.val_mask],
                                                                                  train_out , subgraph.y[subgraph.train_mask],
                                                                                  coexp_adj_mat)
            
            summed_SE_val = float(RMSE_val**2)*num_contrasts_val
            total_num_contrasts_val += num_contrasts_val
            total_summed_SE_val += summed_SE_val

            #relating to testing loss....
            test_out = out[subgraph.test_mask]
            val_out.to(CPU_device)
            RMSE_test, num_contrasts_test = loss_func.RMSE_dotprod_vs_coexp_testval(test_out, subgraph.y[subgraph.test_mask],
                                                                                  train_out , subgraph.y[subgraph.train_mask],
                                                                                  coexp_adj_mat)
            
            summed_SE_test = float((RMSE_test**2))*num_contrasts_test
            total_num_contrasts_test += num_contrasts_test
            total_summed_SE_test += summed_SE_test
            print(f"Mini-Batch Performance | epoch {epoch}, mini-batch {mb_idx}, tr_loss: {RMSE:6f}, val_loss: {RMSE_val:6f}, tst_loss: {RMSE_test:6f}, lr: {learn_rate}")
            with open(log_path, "a") as logout:
                logout.write(f"{epoch}\t{mb_idx}\tMini_Batch_Performance\t{RMSE:6f}\t{RMSE_val:6f}\t{RMSE_test:6f}\t{learn_rate}\n")

        train_loss_aggregated = float(round(np.sqrt(total_summed_SE_training / total_num_contrasts_training), 6))
        val_loss_aggregated = float(round(np.sqrt(total_summed_SE_val / total_num_contrasts_val), 6))
        test_loss_aggregated = float(round(np.sqrt(total_summed_SE_test / total_num_contrasts_test), 6))
        
        print(f"Aggregated Mini-Batch Performance | epoch {epoch}, tr_loss: {train_loss_aggregated:6f}, val_loss: {val_loss_aggregated:6f}, tst_loss: {test_loss_aggregated:6f}, lr: {learn_rate}")
        with open(log_path, "a") as logout:
            logout.write(f"{epoch}\t-\tAggregated_Mini_Batch_Performance\t{train_loss_aggregated:6f}\t{val_loss_aggregated:6f}\t{test_loss_aggregated:6f}\t{learn_rate}\n")
        
        
        scheduler.step(train_loss_aggregated)

        if epoch > 0 and epoch % train_param.inference_interval == 0:
            print(f"epoch: {epoch}| Proceeding with Inference for evaluation")
            infer_out = infer()
            infer_out.to(CPU_device)
            infer_train_out = infer_out[input_graph.train_mask]
            infer_val_out = infer_out[input_graph.val_mask]
            infer_test_out = infer_out[input_graph.test_mask]

            infer_train_RMSE = loss_func.RMSE_dotprod_vs_coexp(infer_train_out, input_graph.y[input_graph.train_mask], coexp_adj_mat)
            infer_val_RMSE, _ = loss_func.RMSE_dotprod_vs_coexp_testval(infer_val_out, input_graph.y[input_graph.val_mask],
                                                                    infer_train_out , input_graph.y[input_graph.train_mask],
                                                                    coexp_adj_mat)
            infer_test_RMSE, _ = loss_func.RMSE_dotprod_vs_coexp_testval(infer_test_out, input_graph.y[input_graph.test_mask],
                                                                    infer_train_out , input_graph.y[input_graph.train_mask],
                                                                    coexp_adj_mat)
            with open(log_path, "a") as logout:
                logout.write(f"{epoch}\t-\tInference_Performance\t{infer_train_RMSE:6f}\t{infer_val_RMSE:6f}\t{infer_test_RMSE:6f}\t{learn_rate}\n")
            print(f"Inference_Performance | epoch {epoch}, tr_loss: {infer_train_RMSE:6f}, val_loss: {infer_val_RMSE:6f}, tst_loss: {infer_test_RMSE:6f}, lr: {learn_rate}")
    
            if infer_train_RMSE < train_param.checkpoint_threshold_loss:
                print(f"Inference training loss for this epoch {infer_train_RMSE:4f} is lower than threshold of {train_param.checkpoint_threshold_loss}. Saving model...")
                model_state_path = os.path.join(model_dump_dir, f"Epoch{epoch}_model_state.pth")
                torch.save(model.state_dict(), model_state_path)
            
            #save embeddings
            if train_param.save_inference_embeddings:
                embeddings_path = os.path.join(infer_dump_dir,f"Epoch{epoch}_emb.pkl")
                with open(embeddings_path, "wb") as fbout:
                    pickle.dump(infer_out, fbout)



             








    

    