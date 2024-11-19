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
#from torch.cuda.amp import autocast
from torch.amp import autocast, GradScaler
from concurrent.futures import ProcessPoolExecutor
import random

from utils import others, models, loss_func



input_graph, coexp_adj_mat = load_input_data(species, train_param.species_data_dir )

def load_input_data(species, species_data_dir):
    print("loading data...")
    input_graph_path = os.
    coexp_adj_mat_path = 
    with open(train_param.input_graph_path, "rb") as finb:
        input_graph = pickle.load(finb)
    print(f"input_graph at {train_param.input_graph_path} loaded")

    with open(train_param.coexp_adj_mat, "rb") as finb:
        coexp_adj_mat = pickle.load(finb)
    print(f"coexp_adj_mat at {train_param.coexp_adj_mat} loaded")
    print("done\n")
    return input_graph, coexp_adj_mat


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
    if train_param.species_data_download_link is not None and not os.path.exists(train_param.input_graph_path):
        print(f"Input data not found. Downloading from {train_param.species_data_download_link}...")
        gdown.download(train_param.species_data_download_link, train_param.species_data_dir)

    #create_outdir
    if not os.path.exists(train_param.output_dir):
        os.makedirs(train_param.output_dir)
    #copy training parameters
    shutil.copy(param_path, os.path.join(train_param.output_dir, "multiplex_train_param.py"))



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
    

    #specify and create subdirs
    training_species_order_list = list(train_param.species_train.keys())
    validation_species_order_list = list(train_param.species_val.keys())
    testing_species_order_list = list(train_param.species_test.keys())

    log_headers= ["Mode", "Type", "Epoch", "Species Training Order", "Species Type", "Species", "Batch", "Training Loss", "Validation Loss" ,"Testing Loss", "Learning Rate"]
    log_filepaths_dict = {}
    log_filepaths_dict["main"] = os.path.join(train_param.output_dir, "Logs", "Main_log.txt")
    
    if not os.path.exists(log_filepaths_dict["main"]):
        os.makedirs(os.path.join(train_param.output_dir, "Logs"))
        with open(log_filepaths_dict["main"], "a") as flogout:
            flogout.write("\t".join(log_headers) + "\n")
    
    embedding_dirpaths_dict = {}
    clustergcn_dirpaths_dict = {}
    
    for species in training_species_order_list + validation_species_order_list + testing_species_order_list:
        
        clustergcn_dirpaths_dict[species] = os.path.join(train_param.clusterGCN_dir, species)
        embedding_dirpaths_dict[species] = os.path.join(train_param.output_dir, "Embeddings" , species)
        
        if not os.path.exists(clustergcn_dirpaths_dict[species]):
            os.makedirs(clustergcn_dirpaths_dict[species])
        if not os.path.exists(embedding_dirpaths_dict[species]):
            os.makedirs(embedding_dirpaths_dict[species])
        
        log_filepaths_dict[species] = os.path.join(train_param.output_dir,  "Logs", species, "Species_Log.txt")
        if not os.path.exists(log_filepaths_dict[species]):
            os.makedirs(os.path.join(train_param.output_dir, "Logs", "species"))
            with open(log_filepaths_dict["species"], "a") as flogout:
                flogout.write("\t".join(log_headers) + "\n")
    
        
    #create subdir to dump model states
    model_dump_dir = os.path.join(train_param.output_dir,"Model_states")
    if not os.path.exists(model_dump_dir):
        os.makedirs(model_dump_dir)
    
    #moving model to GPU device (if applicable) prior to training
    model = model.to(GPU_device)

    #init Gradient scaler for mixed precision training
    if train_param.mode == "CPU":
        scaler = GradScaler('cpu')
    elif train_param.mode == "GPU":
        scaler = GradScaler("cuda")

    #shuffle speices order for first epoch if needed
    if train_param.shuffle_species_order:
        random.seed(train_param.shuffle_seed)
        epoch_training_species_order = training_species_order_list.copy()
        random.shuffle(epoch_training_species_order)
    else:
        epoch_training_species_order = training_species_order_list.copy()

    #main loop to train / eval model :)
    for epoch in range(1,train_param.num_epoch+1):
        summed_SEMBP_train_loss, summed_SEMBP_val_loss, summed_SEMBP_test_loss = 0,0,0
        #training...
        model.train()
        if train_param.shuffle_species_order:
            random.shuffle(epoch_training_species_order)
        #Use ProcessPoolExecutor for background data loading
        with ProcessPoolExecutor(max_workers=1) as executor:
            next_data_future= None
            for species_idx, species in enumerate(epoch_training_species_order):
                species_order = species_idx + 1
                if species_order == 1: # first species
                    input_graph, coexp_adj_mat = load_input_data(species, train_param.species_data_dir )
                else:
                    # Wait for the next species to finish loading
                    input_graph, coexp_adj_mat = next_data_future.result()
                
                # Start loading the next species in the background
                if species_order != len(training_species_order_list): #if not last species
                    next_species = epoch_training_species_order[species_idx + 1]
                    next_data_future = executor.submit(load_input_data, next_species,  train_param.species_data_dir)
                
                # init Cluster-CGN
                cluster_data = ClusterData(input_graph, num_parts  = train_param.clusterGCN_num_parts, save_dir= clustergcn_dirpaths_dict[species], keep_inter_cluster_edges =True)
                loader= ClusterLoader(cluster_data, num_workers = train_param.num_workers, batch_size = train_param.clusterGCN_parts_perbatch, shuffle=True)
                
                # start training model using species data
                summed_MBP_train_loss, summed_MBP_val_loss , summed_MBP_test_loss= 0 , 0, 0
                for mb_idx, subgraph in enumerate(loader):
                    #setup
                    batch_no = mb_idx + 1
                    subgraph = subgraph.to(GPU_device)
                    learn_rate = scheduler.get_last_lr()[-1]
                    optimizer.zero_grad()
                    
                    #forward pass and gradient calc.
                    out = model(subgraph.x,  subgraph.edge_index, subgraph.edge_weight)
                    
                    #calc. MBP training loss....
                    train_out = out[subgraph.train_mask]

                    with autocast(device_type='cuda'if train_param.mode == "GPU" else "cpu", dtype=torch.float16):
                        RMSE = loss_func.RMSE_dotprod_vs_coexp(train_out, subgraph.y[subgraph.train_mask], coexp_adj_mat, CPU_device, GPU_device)
                    scaler.scale(RMSE).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    RMSE = float(RMSE.detach().to(CPU_device))
                    summed_MBP_train_loss += RMSE

                    #detach output after training
                    out = out.detach()
                    out = out.to(CPU_device)
                    train_out = out[subgraph.train_mask.to(CPU_device)]

                    #MBP validation loss calc.
                    val_out = out[subgraph.val_mask.to(CPU_device)]
                    RMSE_val, num_contrasts_val = loss_func.RMSE_dotprod_vs_coexp_testval(val_out, subgraph.y[subgraph.val_mask].to(CPU_device),
                                                                                  train_out , subgraph.y[subgraph.train_mask].to(CPU_device),
                                                                                  coexp_adj_mat)
                    summed_MBP_val_loss += RMSE_val

                    #MBP testing loss calc.
                    test_out = out[subgraph.test_mask.to(CPU_device)]
                    RMSE_test, num_contrasts_test = loss_func.RMSE_dotprod_vs_coexp_testval(test_out, subgraph.y[subgraph.test_mask].to(CPU_device),
                                                                                  train_out , subgraph.y[subgraph.train_mask].to(CPU_device),
                                                                                  coexp_adj_mat)
                    summed_MBP_test_loss += RMSE_test

                    #reporting and logging
                    print(f"Mode: Training\tType: MBP\tEpoch: {epoch}\tOrder: {species_order}\tSpecies: {species}\tBatch: {batch_no}\tTr_loss: {RMSE:5f}\tVal_loss: {RMSE_val:5f}\tTst_loss: {RMSE_test:5f}\tLR: {learn_rate}")
                    with open(log_filepaths_dict[species], "a") as flogout:
                        flogout.write(f"Training\tMini-Batch Performance (MBP)\t{epoch}\t{species_order}\tTraining\t{species}\t{batch_no}\t{RMSE}\t{RMSE_val}\t{RMSE_test}\t{learn_rate}\n")
                    with open(log_filepaths_dict["main"], "a") as flogout:
                        flogout.write(f"Training\tMini-Batch Performance (MBP)\t{epoch}\t{species_order}\tTraining\t{species}\t{batch_no}\t{RMSE}\t{RMSE_val}\t{RMSE_test}\t{learn_rate}\n")
                    
                    #clear cache
                    torch.cuda.empty_cache()

                SEMBP_train_loss = summed_MBP_train_loss / batch_no
                SEMBP_val_loss = summed_MBP_val_loss / batch_no
                SEMBP_test_loss = summed_MBP_test_loss / batch_no
                
                summed_SEMBP_train_loss += SEMBP_train_loss
                summed_SEMBP_val_loss += SEMBP_val_loss
                summed_SEMBP_test_loss += SEMBP_test_loss

                print(f"Mode: Training\tType: SEMBP\tEpoch: {epoch}\tSpecies: {species}\tTr_loss: {SEMBP_train_loss:5f}\tVal_loss: {SEMBP_val_loss:5f}\tTst_loss: {SEMBP_test_loss:5f}")
                with open(log_filepaths_dict[species], "a") as flogout:
                    flogout.write(f"Training\tSpecies-specific Epoch MBP (SEMBP)\t{epoch}\t{species_order}\tTraining\t{species}\t-\t{SEMBP_train_loss}\t{SEMBP_val_loss}\t{SEMBP_test_loss}\t{learn_rate}\n")
                with open(log_filepaths_dict["main"], "a") as flogout:
                    flogout.write(f"Training\tSpecies-specific Epoch MBP (SEMBP)\t{epoch}\t{species_order}\tTraining\t{species}\t-\t{SEMBP_train_loss}\t{SEMBP_val_loss}\t{SEMBP_test_loss}\t{learn_rate}\n")
            
            num_training_species = len(training_species_order_list)
            ASEMBP_train_loss = summed_SEMBP_train_loss / num_training_species
            ASEMBP_val_loss = summed_SEMBP_val_loss / num_training_species
            ASEMBP_test_loss = summed_SEMBP_test_loss / num_training_species

            if epoch % train_param.inference_interval == 0:
                #inference...
                model.eval()
                summed_train_SFBP,  summed_val_SFBP, summed_test_SFBP = 0, 0 , 0 
                with torch.no_grad() and  ProcessPoolExecutor(max_workers=1) as executor:
                    next_data_future= None
                    for species_idx, species in enumerate(training_species_order_list + validation_species_order_list + testing_species_order_list):
                        species_order = species_idx + 1
                        if species_order == 1: # first species
                            input_graph, coexp_adj_mat = load_input_data(species, train_param.species_data_dir )
                        else:
                            # Wait for the next species to finish loading
                            input_graph, coexp_adj_mat = next_data_future.result()
                    
                        # Start loading the next species in the background
                        if species_order != len(training_species_order_list): #if not last species
                            next_species = epoch_training_species_order[species_idx + 1]
                            next_data_future = executor.submit(load_input_data, next_species,  train_param.species_data_dir)
                    
                        # init Cluster-CGN
                        cluster_data = ClusterData(input_graph, num_parts  = train_param.clusterGCN_num_parts, save_dir= clustergcn_dirpaths_dict[species], keep_inter_cluster_edges =True)
                        loader= ClusterLoader(cluster_data, num_workers = train_param.num_workers, batch_size = train_param.clusterGCN_parts_perbatch, shuffle=True) 
                        
                        #run inference
                        rearranged_out = torch.zeros(input_graph.x.size()[0], train_param.decode_kwargs["out_channels"])
                        with autocast(device_type='cuda'if train_param.mode == "GPU" else "cpu", dtype=torch.float16):
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
                        if species in training_species_order_list: # if training species...
                            infer_train_out = infer_out[input_graph.train_mask]
                            infer_val_out = infer_out[input_graph.val_mask]
                            infer_test_out = infer_out[input_graph.test_mask]

                            train_SFBP = loss_func.RMSE_dotprod_vs_coexp(infer_train_out, input_graph.y[input_graph.train_mask], coexp_adj_mat, CPU_device, CPU_device) # both CPU devices
                            val_SFBP, _ = loss_func.RMSE_dotprod_vs_coexp_testval(infer_val_out, input_graph.y[input_graph.val_mask],
                                                                                    infer_train_out , input_graph.y[input_graph.train_mask],
                                                                                    coexp_adj_mat)
                            test_SFBP, _ = loss_func.RMSE_dotprod_vs_coexp_testval(infer_test_out, input_graph.y[input_graph.test_mask],
                                                                                    infer_train_out , input_graph.y[input_graph.train_mask],
                                                                                    coexp_adj_mat)
                            #reporting / logging
                            print(f"Mode: Inference\tType: SFBP\tEpoch: {epoch}\tSpecies Type: Training\tSpecies: {species}\tTr_loss: {train_SFBP:5f}\tVal_loss: {val_SFBP:5f}\tTst_loss: {test_SFBP:5f}")
                            with open(log_filepaths_dict[species], "a") as flogout:
                                flogout.write(f"Inference\tSpecies-specific Full Batch Performance (SFBP)\t{epoch}\t-\tTraining\t{species}\t-\t{train_SFBP}\t{val_SFBP}\t{test_SFBP}\t{learn_rate}\n")
                            with open(log_filepaths_dict["main"], "a") as flogout:
                                flogout.write(f"Inference\tSpecies-specific Full Batch Performance (SFBP)\t{epoch}\t-\tTraining\t{species}\t-\t{train_SFBP}\t{val_SFBP}\t{test_SFBP}\t{learn_rate}\n")

                            summed_train_SFBP += train_SFBP
                            summed_val_SFBP += val_SFBP
                            summed_test_SFBP += test_SFBP

                        else: #for validation / testing species
                            #all nodes are validation / testing nodes.
                            SFBP = loss_func.RMSE_dotprod_vs_coexp(infer_out, input_graph.y, coexp_adj_mat, CPU_device, CPU_device) # both CPU devices
                            
                            #check if validation or testing species
                            if species in validation_species_order_list: # if validation species
                                print(f"Mode: Inference\tType: SFBP\tEpoch: {epoch}\tSpecies Type: Validation\tSpecies: {species}\tTr_loss: -\tVal_loss: {SFBP:5f}\tTst_loss: -")
                                with open(log_filepaths_dict[species], "a") as flogout:
                                    flogout.write(f"Inference\tSpecies-specific Full Batch Performance (SFBP)\t{epoch}\t-\Validation\t{species}\t-\t-\t{SFBP}\t-\t{learn_rate}\n")
                                with open(log_filepaths_dict["main"], "a") as flogout:
                                    flogout.write(f"Inference\tSpecies-specific Full Batch Performance (SFBP)\t{epoch}\t-\Validation\t{species}\t-\t-\t{SFBP}\t-\t{learn_rate}\n")
                                
                                summed_val_SFBP += SFBP
                            
                            else: # if testing species
                                print(f"Mode: Inference\tType: SFBP\tEpoch: {epoch}\tSpecies Type: Validation\tSpecies: {species}\tTr_loss: -\tVal_loss: -\tTst_loss: {SFBP:5f}")
                                with open(log_filepaths_dict[species], "a") as flogout:
                                    flogout.write(f"Inference\tSpecies-specific Full Batch Performance (SFBP)\t{epoch}\t-\Validation\t{species}\t-\t-\t-\t{SFBP}\t{learn_rate}\n")
                                with open(log_filepaths_dict["main"], "a") as flogout:
                                    flogout.write(f"Inference\tSpecies-specific Full Batch Performance (SFBP)\t{epoch}\t-\Validation\t{species}\t-\t-\t-\t{SFBP}\t{learn_rate}\n")
                                
                                summed_test_SFBP += SFBP
                                    #save embeddings
                        if train_param.save_inference_embeddings:
                            embeddings_path = os.path.join(embedding_dirpaths_dict[species],f"Epoch{epoch}_emb.pkl")
                            with open(embeddings_path, "wb") as fbout:
                                pickle.dump(infer_out, fbout)
                        
                    ASFBP_train = summed_train_SFBP / len(training_species_order_list)
                    ASFBP_val = summed_val_SFBP / len(training_species_order_list + validation_species_order_list)
                    ASFBP_test = summed_test_SFBP / len(training_species_order_list + testing_species_order_list)
                    print(f"Mode: Inference\tType: ASFBP\tEpoch: {epoch}\tTr_loss: {ASFBP_train:5f}\tVal_loss: {ASFBP_val:5f}\tTst_loss: {ASFBP_test:5f}")
                    with open(log_filepaths_dict[species], "a") as flogout:
                        flogout.write(f"Inference\tAverage SFBP (ASFBP)\t{epoch}\t-\-\t-\t-\t{ASFBP_train}\t{ASFBP_val}\t{ASFBP_test}\t{learn_rate}\n")
                    with open(log_filepaths_dict["main"], "a") as flogout:
                        flogout.write(f"Inference\tAverage SFBP (ASFBP)\t{epoch}\t-\-\t-\t-\t{ASFBP_train}\t{ASFBP_val}\t{ASFBP_test}\t{learn_rate}\n")
                    model_state_path = os.path.join(model_dump_dir, f"Epoch{epoch}_model_state.pth")
                    torch.save(model.state_dict(), model_state_path)
                    scheduler.step(ASFBP_val)

