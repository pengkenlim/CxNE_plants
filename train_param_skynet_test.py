#Default training parameters

#   System paramters
num_workers = 60    #--------------------------------------------------------- No. workers for some CPU tasks
mode = 'GPU'        #---------------------------------------------------------Options = ['CPU', 'GPU']
precision = "HALF"      #-----------------------------------------------------Half precision to reduce memory overhead. Options = ["FULL", "HALF"]

#   Download links (google drive only). Will only download if input_graph_path and coexp_adj_mat is missing. Put None to ignore
input_graph_link = "https://drive.google.com/uc?id=1F3jdUgdKuCD1wrhcfmRTZh9FtDsALvc4"
coexp_adj_mat_link = "https://drive.google.com/uc?id=1F4eEwFZYyBs9Kf4b-Pa4_Zh20HG01FLB"

#   dir/file paths
input_graph_path = '/home/ken/CxNE_plants/resources/taxid3702/adj_mat_zscore_5percent_data.pkl'        #-------------------------Path to input data. Needs to be a pickled "torch_geometric.data.Data" object describing a homogeneous graph.
coexp_adj_mat = '/home/ken/CxNE_plants/resources/taxid3702/adj_mat_zscore.pkl'      #-------------Path to zscore co-expression strengths between genes in the form of a pickled numpy array.
output_dir = '/mnt/md2/ken/CxNE_plants_data/model_7'     #-------------------------------------Path to directory to output training statistics and save model state

#   training parameters
checkpoint_threshold_loss = 0.5        #-------------------------------------Validation loss threshold to start saving model
num_epoch = 1000       #---------------------------------------------------------Number of epochs to train
optimizer_kwargs = {"lr": 0.01}     #-----------------------------------------Params for ADAM optimizer
scheduler_kwargs = {"factor": 0.316227766,      #-------------------------------------Params for scheduler. Decrease Learning rate by factor if performance does not increase after #epochs = patience
                    "patience":1}        

# clusterGCN parameters
clusterGCN_num_parts = 1000
clusterGCN_parts_perbatch = 100

# full batch inference parameters
inference_interval = 50
inference_replicates = 10
save_inference_embeddings = True

#   model parameters
encode_kwargs = {"dims": [480 , 424, 368, 312],        #-------------------------Params for encoder
                 "out_channels": 256,
                 "batch_norm": True,
                 "batch_norm_aft_last_layer": True,
                 "act_aft_last_layer": True,
                 "act" : "leaky_relu",
                 "act_kwargs": None}

GAT_kwargs = {"dims": [256, 256],       #-------------------------------------Params for GAT
              "out_channels": 256,
              "batch_norm" : True,
              "batch_norm_aft_last_layer": True,
              "act_aft_last_layer": True,
              "act" : "leaky_relu",
              "concat": False,
              "heads": 10,
              "act_kwargs" : None}

decode_kwargs = {"dims": [256 , 128],       #---------------------------------Params for decoder
                 "out_channels": 64,
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
assert precision == "HALF" or precision == "FULL"
