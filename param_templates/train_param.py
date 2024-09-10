#Default training parameters

#   System paramters
num_workers = 12    #--------------------------------------------------------- No. workers for some CPU tasks
mode = 'GPU'        #---------------------------------------------------------Options = ['CPU', 'GPU']

#   dir/file paths
input_graph_path = '/path/to/input_data.pkl'        #-------------------------Path to input data. Needs to be a pickled "torch_geometric.data.Data" object describing a homogeneous graph.
coexp_adj_mat = '/path/to/coexp_adj_mat.pkl'      #-------------Path to zscore co-expression strengths between genes in the form of a pickled numpy array.
output_dir = '/path/to/output_dir/'     #-------------------------------------Path to directory to output training statistics and save model state

#   training parameters
checkpoint_interval = 10        #---------------------------------------------Interval to save model periodically
checkpoint_threshold_loss = 0.35        #-------------------------------------Validation loss threshold to start saving model
precision = "HALF"      #-----------------------------------------------------Half precision to reduce memory overhead. Options = ["FULL", "HALF"]
num_epoch = 1_000       #-----------------------------------------------------Number of epochs to train
optimizer_kwargs = {"lr": 0.01}     #-----------------------------------------Params for ADAM optimizer
scheduler_kwargs = {"factor": 0.5,      #-------------------------------------Params for scheduler. Decrease Learning rate by factor if performance does not increase after #epochs = patience
                    "patience":10, 
                    "verbose":True}        
datasampler_kwargs = {"num_steps": 30, #---------------------------------------Params for graphsaint random walk graph sampler to generate subgraphs.
                      "batch_size" : 1_000, "walk_length" : 3, "sample_coverage":1 }

#   model parameters
encode_kwargs = {"dims": [480 , 360 , 240 ],        #-------------------------Params for encoder
                 "out_channels": 128, 
                 "batch_norm": True,
                 "batch_norm_aft_last_layer": True,
                 "act_aft_last_layer": True,
                 "act" : "leaky_relu",
                 "act_kwargs": None}

GAT_kwargs = {"dims": [128, 128],       #-------------------------------------Params for GAT
              "out_channels": 128,
              "batch_norm" : True,
              "batch_norm_aft_last_layer": True,
              "act_aft_last_layer": True,
              "act" : "leaky_relu",
              "concat": False,
              "heads": 10,
              "act_kwargs" : None}

decode_kwargs = {"dims": [128 , 128],       #---------------------------------Params for decoder
                 "out_channels": 64,
                 "batch_norm": True,
                 "batch_norm_aft_last_layer": False,
                 "act_aft_last_layer": False,
                 "act" : "leaky_relu",
                 "act_kwargs" : None}

CxNE_kwargs = {"encode_kwargs": encode_kwargs,
              "decode_kwargs": decode_kwargs,
              "GAT_kwargs": GAT_kwargs}

