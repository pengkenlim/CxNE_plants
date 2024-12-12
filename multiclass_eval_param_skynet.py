species = "taxid3702"
datasetprefix = "ESM3B_concat_RP11_E500"
labelname = "multi_class_60ptolerance"

#trainng params
batch_size =  512 
learning_rate = 1e-4 
num_epochs = 500
dropout_rate = 0.1 

#seeding
random_seed = 42
no_fold_ds1_only = True # only DS1 and first fold

#save checkpoint,
save_checkpoint = True
checkpoint_interval = 5


#   Wandblogging
project = "CxNE_Eval_SPM"
name = f"Jason_{species}_{datasetprefix}_{labelname}_dpo01_BS{batch_size}_LR1e-4_3L"
entity="crowdsourced_bioinformatics"

#defining output dir
outputdir = f"/mnt/md2/ken/CxNE_plants_data/evaluate_downstream/{name}/"

# k-fold 
k= 5

# Parameters
MLP_kwargs = {"dims" : [2656 , 332, 42],
"out_channels" : 6, #Must be the same as number of clases
"norm_type" : "batch_norm",
"norm_aft_last_layer" : False,
"act_aft_last_layer" : False,
"act" : "leaky_relu",
"act_kwargs" : None,
"dropout_rate" : dropout_rate} 
    
#defining input dir 
speciesdir = f"/mnt/md2/ken/CxNE_plants_data/species_data/{species}/"
datasetdir = speciesdir + f"datasets/{datasetprefix}"
labeldir = speciesdir + f"labels/{labelname}"
