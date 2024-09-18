#Default training parameters

#   System paramters
num_workers = 9    #--------------------------------------------------------- No. workers for some CPU tasks
mode = 'GPU'        #---------------------------------------------------------Options = ['CPU', 'GPU']
precision = "HALF"      #-----------------------------------------------------Half precision to reduce memory overhead. Options = ["FULL", "HALF"]

#   Download links (google drive only). Will only download if input_graph_path and coexp_adj_mat is missing. Put None to ignore
input_graph_link = "https://drive.google.com/uc?id=1FLAeMS1EH989RRDQGKA2aAW_U1MLG-k2"
coexp_adj_mat_link = None #"https://drive.google.com/uc?id=1F4eEwFZYyBs9Kf4b-Pa4_Zh20HG01FLB"

#   dir/file paths
input_graph_path = '/workspace/data/taxid3702/adj_mat_zscore_15percent_data.pkl'        #-------------------------Path to input data. Needs to be a pickled "torch_geometric.data.Data" object describing a homogeneous graph.
coexp_adj_mat = '/workspace/data/taxid3702/adj_mat_zscore.pkl'      #-------------Path to zscore co-expression strengths between genes in the form of a pickled numpy array.
output_dir = '/workspace/data/model_24_relu_loss_05psubgraph_15density'     #-------------------------------------Path to directory to output training statistics and save model state

#   training parameters
checkpoint_threshold_loss = 0.6        #-------------------------------------Validation loss threshold to start saving model
num_epoch = 205       #---------------------------------------------------------Number of epochs to train
optimizer_kwargs = {"lr": 0.01}     #-----------------------------------------Params for ADAM optimizer
scheduler_kwargs = {"factor": 0.5,      #-------------------------------------Params for scheduler. Decrease Learning rate by factor if performance does not increase after #inferences = patience
                    "patience":0}        #the patience here refer to the number of inferences without validation loss decrease

# clusterGCN parameters
clusterGCN_num_parts = 1000
clusterGCN_parts_perbatch = 50

# full batch inference parameters
inference_interval = 50
inference_replicates = 10
save_inference_embeddings = True

#   model parameters
encode_kwargs = {"dims": [480 , 410, 340, 270, 200],        #-------------------------Params for encoder 
                 "out_channels": 64, 
                 "batch_norm": True,
                 "batch_norm_aft_last_layer": True,
                 "act_aft_last_layer": True,
                 "act" : "leaky_relu",
                 "act_kwargs": None}

GAT_kwargs = {"dims": [64, 64],       #-------------------------------------Params for GAT
              "out_channels": 64,
              "batch_norm" : True,
              "batch_norm_aft_last_layer": True, #True
              "act_aft_last_layer": True, #True
              "act" : "leaky_relu",
              "concat": False,
              "heads": 10,
              "act_kwargs" : None}

decode_kwargs = {"dims": [64 , 64, 64],       #---------------------------------Params for decoder[64 , 64, 64]
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
