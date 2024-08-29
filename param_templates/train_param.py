#Default training parameters

#   System paramters
num_workers = 12    #--------------------------------------------------------- No. workers for some CPU tasks
mode = 'GPU'        #---------------------------------------------------------Options = ['CPU', 'GPU']

#   dir/file paths
input_graph_path = '/path/to/input_data.pkl'        #-------------------------Path to input data. Needs to be a pickled "torch_geometric.data.Data" object describing a homogeneous graph.
scaled_coexp_adj_mat = '/path/to/scaled_coexp_adj_mat.pkl'      #-------------Path to scaled co-expression strengths between genes in the form of a pickled numpy array.
output_dir = '/path/to/output_dir/'     #-------------------------------------Path to directory to output training statistics and save model state

#   training parameters
checkpoint_interval = 10        #---------------------------------------------Interval to save model periodically
checkpoint_threshold_loss = 0.35        #-------------------------------------Validation loss threshold to start saving model
precision = "HALF"      #-----------------------------------------------------Half precision to reduce memory overhead. Options = ["FULL", "HALF"]
epoch = 1_000       #---------------------------------------------------------Number of epochs to train
optimizer_kwargs = {"lr": 0.01}
scheduler_kwargs = {"factor": 0.5, "patience":10, "verbose":False}
datasampler_kwargs = {"num_steps": 6, "batch_size" : 5_000, "walk_length" : 3 }

#   model parameters
encode_kwargs = {"dims": [480 , 360 , 240 , 128] }
assert 
decode_kwargs = {"dims": [128 , 256, 128] }
GNN_kwargs = {"encode_kwargs": encode_kwargs,
                "act": "leaky_relu",}