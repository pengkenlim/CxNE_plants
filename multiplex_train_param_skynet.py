#Default training parameters

#   Wandblogging
project = "CxNE_Viridiplantae"
name = "Training_1"

#   System paramters
num_workers = 32    #--------------------------------------------------------- No. workers for some CPU tasks
mode = 'CPU'        #---------------------------------------------------------Options = ['CPU', 'GPU']

#   dir/file paths
input_graph_filename = "adj_mat_zscore_20percent_data.pkl"
coexp_adj_mat_filename = "adj_mat_zscore.pkl"
species_data_dir = "/mnt/md2/ken/CxNE_plants_data/species_data/"
clusterGCN_dir = "/mnt/md2/ken/CxNE_plants_data/clusterGCN"
intermediate_data_dir = "/mnt/md2/ken/CxNE_plants_data/intermediate_data"
overwrite_intermediate_data = False
#species_train = {"taxid3702": "Arabidopsis thaliana", "taxid29760":"Vitis vinifera", "taxid3694": "Populus trichocarpa"}
#species_val = {"taxid59689": "Arabidopsis lyrata"}
#species_test = {"taxid81970": "Arabidopsis halleri"}
species_train = {"taxid3702_old": "Arabidopsis thaliana","taxid59689_old": "Arabidopsis lyrata"}
species_val = {}
species_test = {"taxid81970_old": "Arabidopsis halleri"}
output_dir = '/mnt/md2/ken/CxNE_plants_data/multiplex_models/model_9'     #-------------------------------------Path to directory to output training statistics and save model state



#   data_downloading (gdrive using gdown)
species_data_download_link = None

#   species order shuffling
shuffle_species_order =True
shuffle_seed = 42

#   training parameters
num_epoch = 30       #---------------------------------------------------------Number of epochs to train
optimizer_kwargs = {"lr": 0.01}     #-----------------------------------------Params for ADAM optimizer
scheduler_kwargs = {"factor": 0.5,      #-------------------------------------Params for scheduler. Decrease Learning rate by factor if performance does not increase after #inferences = patience
                    "patience":0}        #the patience here refer to the number of inferences without validation loss decrease

# clusterGCN parameters
clusterGCN_num_parts = 1000
clusterGCN_parts_perbatch = 50

# full batch inference parameters
inference_interval = 2
inference_replicates = 2
save_inference_embeddings = True


#   model parameters
encode_kwargs = {"dims": [480],        #-------------------------Params for encoder 
                 "out_channels": 4, 
                 "batch_norm": True,
                 "batch_norm_aft_last_layer": True,
                 "act_aft_last_layer": True,
                 "act" : "leaky_relu",
                 "act_kwargs": None}

GAT_kwargs = {"dims": [4, 4],       #-------------------------------------Params for GAT
              "out_channels": 4,
              "batch_norm" : True,
              "batch_norm_aft_last_layer": True, 
              "act_aft_last_layer": True, 
              "act" : "leaky_relu",
              "concat": False,
              "heads": 1,
              "act_kwargs" : None}

decode_kwargs = {"dims": [4],       #---------------------------------Params for decoder[64 , 64, 64]
                 "out_channels": 4,
                 "batch_norm": True,
                 "batch_norm_aft_last_layer": False,
                 "act_aft_last_layer": False,
                 "act" : "leaky_relu",
                 "act_kwargs" : None}

CxNE_kwargs = {"encode_kwargs": encode_kwargs,
              "decode_kwargs": decode_kwargs,
              "GAT_kwargs": GAT_kwargs}

# asserssions to check if arguments of layers makes sense

assert encode_kwargs["out_channels"] == GAT_kwargs["dims"][0]
assert GAT_kwargs["out_channels"] == decode_kwargs["dims"][0]
assert mode == "CPU" or mode == "GPU"
