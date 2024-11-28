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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import random
import joblib
import wandb
import time
from utils import others, models, loss_func
import gc


def load_input_data(species):
    print(f"pre-loading data... for {species}")
    input_graph_path = os.path.join(train_param.species_data_dir, species,  train_param.input_graph_filename)
    coexp_adj_mat_path = os.path.join(train_param.species_data_dir, species, train_param.coexp_adj_mat_filename)
    with open(input_graph_path, "rb") as finb:
        input_graph = pickle.load(finb)
    print(f"input_graph at {input_graph_path} loaded")

    with open(coexp_adj_mat_path, "rb") as finb:
        coexp_adj_mat = pickle.load(finb)
    print(f"coexp_adj_mat at {coexp_adj_mat_path} loaded")
    return input_graph, coexp_adj_mat

def load_input_data_for_training(species):
    print(f"pre-loading data... for {species}")
    coexp_adj_mat_path = os.path.join(train_param.intermediate_data_dir, species, "coexp_adj_mat.joblib")
    coexp_adj_mat = joblib.load(coexp_adj_mat_path)
    loader_path = os.path.join(train_param.intermediate_data_dir, species, "loader.joblib")
    loader = joblib.load(loader_path)
    return loader, coexp_adj_mat

def load_input_data_for_inference(species):
    print(f"pre-loading data... for {species}")
    coexp_adj_mat_path = os.path.join(train_param.intermediate_data_dir, species, "coexp_adj_mat.joblib")
    coexp_adj_mat = joblib.load(coexp_adj_mat_path)
    loader_path = os.path.join(train_param.intermediate_data_dir, species, "loader.joblib")
    loader = joblib.load(loader_path)
    input_graph_node_prop_path = os.path.join(train_param.intermediate_data_dir, species,"input_graph_node_prop.joblib")
    input_graph_node_prop = joblib.load(input_graph_node_prop_path)
    return loader, coexp_adj_mat, input_graph_node_prop

if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="CxNE_plants/main/train.py. Train a Graph Neural Network-based Model to learn Gene co-expression embeddings (CxNE).")
    
    parser.add_argument("-p", "--param_path",  type=str ,required = True,
                        help= "File path to parameters needed to run train.py.")
    
    #load params
    args=parser.parse_args()
    param_path = args.param_path
    train_param = others.parse_parameters(param_path)
    #train_param = others.parse_parameters("/home/ken/CxNE_plants/multiplex_train_param_skynet.py")
    
    #downloading data
    if train_param.species_data_download_link is not None and not os.path.exists(train_param.input_graph_path):
        print(f"Input data not found. Downloading from {train_param.species_data_download_link}...")
        gdown.download(train_param.species_data_download_link, train_param.species_data_dir)

    #create_outdir
    if not os.path.exists(train_param.output_dir):
        try:
            os.makedirs(train_param.output_dir)
        except:
            pass
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

    #wandb init
    wandb.init(
    # set the wandb project where this run will be logged
    project= train_param.project,

    name= train_param.name ,

    # track hyperparameters and run metadata
    config={attr: getattr(train_param, attr)
        for attr in dir(train_param)
        if not attr.startswith("__") and not callable(getattr(train_param, attr))}
    )
    

    #specify and create subdirs
    training_species_order_list = list(train_param.species_train.keys())
    validation_species_order_list = list(train_param.species_val.keys())
    testing_species_order_list = list(train_param.species_test.keys())

    log_headers= ["Mode", "Type", "Epoch", "Species Training Order", "Species Type", "Species", "Batch", "Training Loss", "Validation Loss" ,"Testing Loss", "Learning Rate"]
    log_filepaths_dict = {}
    log_filepaths_dict["main"] = os.path.join(train_param.output_dir, "Logs", "Main_log.txt")
    
    if not os.path.exists(log_filepaths_dict["main"]):
        try:
            os.makedirs(os.path.join(train_param.output_dir, "Logs"))
        except:
            pass
        with open(log_filepaths_dict["main"], "a") as flogout:
            flogout.write("\t".join(log_headers) + "\n")
    
    embedding_dirpaths_dict = {}
    clustergcn_dirpaths_dict = {}
    intermediate_data_dirpaths_dict = {}
    
    for species in training_species_order_list + validation_species_order_list + testing_species_order_list:
        
        clustergcn_dirpaths_dict[species] = os.path.join(train_param.clusterGCN_dir, species)
        embedding_dirpaths_dict[species] = os.path.join(train_param.output_dir, "Embeddings" , species)
        intermediate_data_dirpaths_dict[species] = os.path.join(train_param.intermediate_data_dir,  species)
        
        if not os.path.exists(clustergcn_dirpaths_dict[species]):
            try:
                os.makedirs(clustergcn_dirpaths_dict[species])
            except:
                pass
        if not os.path.exists(embedding_dirpaths_dict[species]):
            try:
                os.makedirs(embedding_dirpaths_dict[species])
            except:
                pass
        if not os.path.exists(intermediate_data_dirpaths_dict[species]):
            try:
                os.makedirs(intermediate_data_dirpaths_dict[species])
            except:
                pass
        
        log_filepaths_dict[species] = os.path.join(train_param.output_dir,  "Logs", species, "Species_Log.txt")
        if not os.path.exists(log_filepaths_dict[species]):
            try:
                os.makedirs(os.path.join(train_param.output_dir, "Logs", species))
            except:
                pass
            with open(log_filepaths_dict[species], "a") as flogout:
                flogout.write("\t".join(log_headers) + "\n")
    
        
    #create subdir to dump model states
    model_dump_dir = os.path.join(train_param.output_dir,"Model_states")
    if not os.path.exists(model_dump_dir):
        try:
            os.makedirs(model_dump_dir)
        except:
            pass
    
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

    #if need overwrite_intermediate_data is False, only prepreocess species without intermediate directory 
    if train_param.overwrite_intermediate_data:
        combined_species = training_species_order_list + validation_species_order_list + testing_species_order_list
    else:
        combined_species = []
        for species in training_species_order_list + validation_species_order_list + testing_species_order_list:
            missing=True
            if os.path.exists(os.path.join(intermediate_data_dirpaths_dict[species], "loader.joblib")):
                if os.path.exists(os.path.join(intermediate_data_dirpaths_dict[species], "coexp_adj_mat.joblib")):
                    if os.path.exists(os.path.join(intermediate_data_dirpaths_dict[species], "input_graph_node_prop.joblib")):
                        missing = False
            if missing:
                combined_species.append(species)

    #loop to preprocess data
    if len(combined_species) >0:
        print(f"Preprocessing data: Graph partitioning, storing intermediate files into {train_param.intermediate_data_dir} for repeated loading during training & inference...")
        with ThreadPoolExecutor(max_workers=2) as preprocess_executor:
            next_data_future_preprocess = None
            for species_idx, species in enumerate(combined_species):
                species_order = species_idx + 1
                if species_order == 1: # first species
                    input_graph, coexp_adj_mat = load_input_data(species)
                else:
                    # Wait for the next species to finish loading
                    input_graph, coexp_adj_mat = next_data_future_preprocess.result()
                # Start loading the next species in the background
                if species_order != len(combined_species): #if not last species
                    next_species = combined_species[species_idx + 1]
                    next_data_future_preprocess = preprocess_executor.submit(load_input_data, next_species)
                
                # init Cluster-CGN
                cluster_data = ClusterData(input_graph, num_parts  = train_param.clusterGCN_num_parts, save_dir= clustergcn_dirpaths_dict[species], keep_inter_cluster_edges =True)
                loader= ClusterLoader(cluster_data, num_workers = train_param.num_workers, batch_size = train_param.clusterGCN_parts_perbatch, shuffle=True)
                
                # generate input_graph_node_prop
                input_graph_node_prop ={"y": input_graph.y,
                                        "train_mask":input_graph.train_mask,
                                        "val_mask":input_graph.val_mask,
                                        "test_mask":input_graph.test_mask}

                # save data
                loader_path = os.path.join(intermediate_data_dirpaths_dict[species], "loader.joblib")
                joblib.dump(loader, loader_path, compress=('lz4', 0)) #basically no compression at all. 

                coexp_adj_mat_path = os.path.join(intermediate_data_dirpaths_dict[species], "coexp_adj_mat.joblib")
                joblib.dump(coexp_adj_mat, coexp_adj_mat_path, compress=('lz4', 0)) #basically no compression at all. 

                input_graph_node_prop_path = os.path.join(intermediate_data_dirpaths_dict[species], "input_graph_node_prop.joblib")
                joblib.dump(input_graph_node_prop, input_graph_node_prop_path, compress=('lz4', 0)) #basically no compression at all. 
                print(f"{species} preprocessing done.")
            print("Data preprocessing complete.")
    else:
        print("Overwrite disabled by user and there are no intermediate data to preprocess.")

    print("Starting model training...")
    #main loop to train / eval model :)
    combined_species = training_species_order_list + validation_species_order_list + testing_species_order_list
    for epoch in range(1,train_param.num_epoch+1):
        epoch_performance_dict = {"epoch": epoch}
        epoch_performance_dict["losses"] = {"training_mode":{}}

        summed_AMBP_train_loss, summed_AMBP_val_loss, summed_AMBP_test_loss = 0,0,0
        #training...
        model.train()
        if train_param.shuffle_species_order:
            random.shuffle(epoch_training_species_order)
        
        #Use ProcessPoolExecutor for background data loading
        
        epoch_performance_dict["training_order"] = {species: idx +1 for idx , species in enumerate(epoch_training_species_order)}
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            epoch_performance_dict["losses"]["training_mode"]["AMBP"] = {}
            next_data_future= None
            for species_idx, species in enumerate(epoch_training_species_order):
                species_order = species_idx + 1
                if species_order == 1: # first species
                    loader, coexp_adj_mat = load_input_data_for_training(species)
                else:
                    # Wait for the next species to finish loading
                    loader, coexp_adj_mat = next_data_future.result()
                    del next_data_future
                    gc.collect()
                
                # Start loading the next species in the background
                if species_order != len(training_species_order_list): #if not last species
                    next_species = epoch_training_species_order[species_idx + 1]
                    next_data_future = executor.submit(load_input_data_for_training, next_species)
                
                # start training model using species data
                summed_MBP_train_loss, summed_MBP_val_loss , summed_MBP_test_loss= 0 , 0, 0
                
                for mb_idx, subgraph in enumerate(loader):
                    with autocast(device_type='cuda'if train_param.mode == "GPU" else "cpu", dtype=torch.float16):
                        #setup
                        batch_no = mb_idx + 1
                        subgraph = subgraph.to(GPU_device)
                        learn_rate = scheduler.get_last_lr()[-1]
                        optimizer.zero_grad()
                        
                        #forward pass and gradient calc.
                        out = model(subgraph.x,  subgraph.edge_index, subgraph.edge_weight)
                        
                        #calc. MBP training loss....
                        train_out = out[subgraph.train_mask]
                        RMSE = loss_func.RMSE_dotprod_vs_coexp(train_out, subgraph.y[subgraph.train_mask], coexp_adj_mat, GPU_device)

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
                del loader
                del coexp_adj_mat
                gc.collect()

                AMBP_train_loss = summed_MBP_train_loss / batch_no
                AMBP_val_loss = summed_MBP_val_loss / batch_no
                AMBP_test_loss = summed_MBP_test_loss / batch_no
                
                summed_AMBP_train_loss += AMBP_train_loss
                summed_AMBP_val_loss += AMBP_val_loss
                summed_AMBP_test_loss += AMBP_test_loss

                print(f"Mode: Training\tType: AMBP\tEpoch: {epoch}\tSpecies: {species}\tTr_loss: {AMBP_train_loss:5f}\tVal_loss: {AMBP_val_loss:5f}\tTst_loss: {AMBP_test_loss:5f}")
                with open(log_filepaths_dict[species], "a") as flogout:
                    flogout.write(f"Training\tAverage MBP (AMBP)\t{epoch}\t{species_order}\tTraining\t{species}\t-\t{AMBP_train_loss}\t{AMBP_val_loss}\t{AMBP_test_loss}\t{learn_rate}\n")
                with open(log_filepaths_dict["main"], "a") as flogout:
                    flogout.write(f"Training\tAverage MBP (AMBP)\t{epoch}\t{species_order}\tTraining\t{species}\t-\t{AMBP_train_loss}\t{AMBP_val_loss}\t{AMBP_test_loss}\t{learn_rate}\n")
                
                epoch_performance_dict["losses"]["training_mode"]["AMBP"][species] = {"tr_loss": AMBP_train_loss,
                                                                                        "val_loss": AMBP_val_loss,
                                                                                        "tst_loss": AMBP_test_loss}
            
            num_training_species = len(training_species_order_list)
            
            Average_AMBP_train_loss = summed_AMBP_train_loss / num_training_species
            Average_AMBP_val_loss = summed_AMBP_val_loss / num_training_species
            Average_AMBP_test_loss = summed_AMBP_test_loss / num_training_species
            print(f"Mode: Training\tType: Average AMBP\tEpoch: {epoch}\tSpecies: -\tTr_loss: {Average_AMBP_train_loss:5f}\tVal_loss: {Average_AMBP_val_loss:5f}\tTst_loss: {Average_AMBP_test_loss:5f}")
            with open(log_filepaths_dict[species], "a") as flogout:
                flogout.write(f"Training\tAverage AMBP\t{epoch}\t-\t-\t-\t-\t{Average_AMBP_train_loss}\t{Average_AMBP_val_loss}\t{Average_AMBP_test_loss}\t{learn_rate}\n")
            with open(log_filepaths_dict["main"], "a") as flogout:
                flogout.write(f"Training\tAverage AMBP\t{epoch}\t-\t-\t-\t-\t{Average_AMBP_train_loss}\t{Average_AMBP_val_loss}\t{Average_AMBP_test_loss}\t{learn_rate}\n")
            
            epoch_performance_dict["losses"]["training_mode"]["Average AMBP"] = {"tr_loss": Average_AMBP_train_loss,
                                                                                        "val_loss": Average_AMBP_val_loss,
                                                                                        "tst_loss": Average_AMBP_test_loss}
            epoch_performance_dict["learning_rate"] = learn_rate

        if epoch % train_param.inference_interval == 0:
            #inference...
            model.eval()
            summed_train_FBP,  summed_val_FBP, summed_test_FBP = 0, 0 , 0 
            epoch_performance_dict["losses"]["inference_mode"] = {}
            epoch_performance_dict["losses"]["inference_mode"]["FBP"] = {} 
            with torch.no_grad(): 
                with ThreadPoolExecutor(max_workers=2) as executor_inference:
                    next_data_future_inference= None
                    
                    for species_idx, species in enumerate(combined_species):
                        species_order = species_idx + 1
                        if species_order == 1: # first species
                            loader, coexp_adj_mat, input_graph_node_prop = load_input_data_for_inference(species )
                        else:
                            # Wait for the next species to finish loading
                            loader, coexp_adj_mat, input_graph_node_prop = next_data_future_inference.result()
                            del next_data_future_inference
                            gc.collect()
                        
                        # Start loading the next species in the background
                        if species_order != len(combined_species): #if not last species
                            next_species = combined_species[species_idx + 1]
                            next_data_future_inference = executor_inference.submit(load_input_data_for_inference, next_species)
                            
                        #run inference
                        rearranged_out = torch.zeros(len(input_graph_node_prop["y"]), train_param.decode_kwargs["out_channels"])
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
                            infer_train_out = infer_out[input_graph_node_prop["train_mask"]]
                            infer_val_out = infer_out[input_graph_node_prop["val_mask"]]
                            infer_test_out = infer_out[input_graph_node_prop["test_mask"]]

                            train_FBP = loss_func.RMSE_dotprod_vs_coexp(infer_train_out, input_graph_node_prop["y"][input_graph_node_prop["train_mask"]], coexp_adj_mat, CPU_device) # use CPU device
                            val_FBP, _ = loss_func.RMSE_dotprod_vs_coexp_testval(infer_val_out, input_graph_node_prop["y"][input_graph_node_prop["val_mask"]],
                                                                                        infer_train_out , input_graph_node_prop["y"][input_graph_node_prop["train_mask"]],
                                                                                        coexp_adj_mat)
                            test_FBP, _ = loss_func.RMSE_dotprod_vs_coexp_testval(infer_test_out, input_graph_node_prop["y"][input_graph_node_prop["test_mask"]],
                                                                                        infer_train_out , input_graph_node_prop["y"][input_graph_node_prop["train_mask"]],
                                                                                        coexp_adj_mat)
                            #reporting / logging
                            print(f"Mode: Inference\tType: FBP\tEpoch: {epoch}\tSpecies Type: Training\tSpecies: {species}\tTr_loss: {train_FBP:5f}\tVal_loss: {val_FBP:5f}\tTst_loss: {test_FBP:5f}")
                            with open(log_filepaths_dict[species], "a") as flogout:
                                flogout.write(f"Inference\tFull Batch Performance (FBP)\t{epoch}\t-\tTraining\t{species}\t-\t{train_FBP}\t{val_FBP}\t{test_FBP}\t{learn_rate}\n")
                            with open(log_filepaths_dict["main"], "a") as flogout:
                                flogout.write(f"Inference\tFull Batch Performance (FBP)\t{epoch}\t-\tTraining\t{species}\t-\t{train_FBP}\t{val_FBP}\t{test_FBP}\t{learn_rate}\n")
                            
                            epoch_performance_dict["losses"]["inference_mode"]["FBP"][species] = {"tr_loss": train_FBP,
                                                                                        "val_loss": val_FBP,
                                                                                        "tst_loss": test_FBP,
                                                                                        "species_type": "Training"}
                            
                            summed_train_FBP += train_FBP
                            summed_val_FBP += val_FBP
                            summed_test_FBP += test_FBP

                        else: #for validation / testing species
                            #all nodes are validation / testing nodes.
                            FBP = loss_func.RMSE_dotprod_vs_coexp(infer_out, input_graph_node_prop["y"], coexp_adj_mat, CPU_device) # use CPU device
                                
                            #check if validation or testing species
                            if species in validation_species_order_list: # if validation species
                                print(f"Mode: Inference\tType: FBP\tEpoch: {epoch}\tSpecies Type: Validation\tSpecies: {species}\tTr_loss: -\tVal_loss: {FBP:5f}\tTst_loss: -")
                                with open(log_filepaths_dict[species], "a") as flogout:
                                    flogout.write(f"Inference\tFull Batch Performance (FBP)\t{epoch}\t-\Validation\t{species}\t-\t-\t{FBP}\t-\t{learn_rate}\n")
                                with open(log_filepaths_dict["main"], "a") as flogout:
                                    flogout.write(f"Inference\tFull Batch Performance (FBP)\t{epoch}\t-\Validation\t{species}\t-\t-\t{FBP}\t-\t{learn_rate}\n")
                                
                                epoch_performance_dict["losses"]["inference_mode"]["FBP"][species] = {"val_loss": FBP,
                                                                                        "species_type": "Validation"}
                            
                                summed_val_FBP += FBP

                            else: # if testing species
                                print(f"Mode: Inference\tType: FBP\tEpoch: {epoch}\tSpecies Type: Testing\tSpecies: {species}\tTr_loss: -\tVal_loss: -\tTst_loss: {FBP:5f}")
                                with open(log_filepaths_dict[species], "a") as flogout:
                                    flogout.write(f"Inference\tFull Batch Performance (FBP)\t{epoch}\t-\Testing\t{species}\t-\t-\t-\t{FBP}\t{learn_rate}\n")
                                with open(log_filepaths_dict["main"], "a") as flogout:
                                    flogout.write(f"Inference\tFull Batch Performance (FBP)\t{epoch}\t-\Testing\t{species}\t-\t-\t-\t{FBP}\t{learn_rate}\n")
                                
                                epoch_performance_dict["losses"]["inference_mode"]["FBP"][species] = {"tst_loss": FBP,
                                                                                        "species_type": "Testing"}
                                    
                                summed_test_FBP += FBP
                        #save embeddings
                        if train_param.save_inference_embeddings:
                            embeddings_path = os.path.join(embedding_dirpaths_dict[species],f"Epoch{epoch}_emb.pkl")
                            with open(embeddings_path, "wb") as fbout:
                                pickle.dump(infer_out, fbout)
                        del loader
                        del coexp_adj_mat
                        del input_graph_node_prop
                        gc.collect()
                    
            Average_FBP_train = summed_train_FBP / len(training_species_order_list)
            Average_FBP_val = summed_val_FBP / len(training_species_order_list + validation_species_order_list)
            Average_FBP_test = summed_test_FBP / len(training_species_order_list + testing_species_order_list)

            print(f"Mode: Inference\tType: AFBP\tEpoch: {epoch}\tTr_loss: {Average_FBP_train:5f}\tVal_loss: {Average_FBP_val:5f}\tTst_loss: {Average_FBP_test:5f}")
            with open(log_filepaths_dict[species], "a") as flogout:
                flogout.write(f"Inference\tAverage FBP\t{epoch}\t-\-\t-\t-\t{Average_FBP_train}\t{Average_FBP_val}\t{Average_FBP_test}\t{learn_rate}\n")
            with open(log_filepaths_dict["main"], "a") as flogout:
                flogout.write(f"Inference\tAverage FBP\t{epoch}\t-\-\t-\t-\t{Average_FBP_train}\t{Average_FBP_val}\t{Average_FBP_test}\t{learn_rate}\n")
            
            epoch_performance_dict["losses"]["inference_mode"]["Average FBP"]= {"tr_loss": Average_FBP_train,
                                                                                        "val_loss": Average_FBP_val,
                                                                                        "tst_loss": Average_FBP_test}

            model_state_path = os.path.join(model_dump_dir, f"Epoch{epoch}_model_state.pth")
            torch.save(model.state_dict(), model_state_path)
            scheduler.step(Average_FBP_val)

        wandb.log(epoch_performance_dict)
    wandb.finish()

