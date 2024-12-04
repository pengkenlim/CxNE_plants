#setting sys.path for importing modules
import os
import sys

if __name__ == "__main__":
         abspath= __file__
         parent_module= "/".join(abspath.split("/")[:-2])
         sys.path.insert(0, parent_module)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn.resolver import activation_resolver
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
import torch.nn.functional as F
import pickle
import shutil
import wandb
import argparse

from utils import others


# Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.clone().to(dtype=torch.float32)       
        self.labels = labels.clone().to(dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#return a Multi layer perceptron (MLP)
def return_mlp(dims, out_channels, norm_type, norm_aft_last_layer, act_aft_last_layer, act, act_kwargs, dropout_rate=None):
    """
    Builds an MLP (Multi-Layer Perceptron) with configurable normalization, activation, and dropout.

    Args:
        dims (list): Dimensions of the hidden layers.
        out_channels (int): Number of output channels.
        norm_type (str or None): Normalization type ('batch_norm', 'layer_norm', or None).
        norm_aft_last_layer (bool): Apply normalization after the last layer.
        act_aft_last_layer (bool): Apply activation after the last layer.
        act (str): Activation function name (resolved by `activation_resolver`).
        act_kwargs (dict or None): Additional kwargs for the activation function.
        dropout_rate (float or None): Dropout rate for regularization.

    Returns:
        nn.Sequential: Configured MLP model.
    """
    mlp = nn.Sequential()
    for layer, in_dim in enumerate(dims):
        if layer < len(dims) - 1:  # If before the last layer
            out_dim = dims[layer + 1]  # Out_dim is the in_dim of the subsequent layer
            # Add linear layer
            mlp.append(nn.Linear(in_dim, out_dim))
            # Add normalization if specified
            if norm_type == 'batch_norm':
                mlp.append(nn.BatchNorm1d(out_dim, track_running_stats=False))
            elif norm_type == 'layer_norm':
                mlp.append(nn.LayerNorm(out_dim))
            # Add activation function
            mlp.append(activation_resolver(act, **(act_kwargs or {})))
            # Add dropout if specified
            if dropout_rate is not None:
                mlp.append(nn.Dropout(dropout_rate))
        else:  # Last layer
            mlp.append(nn.Linear(in_dim, out_channels))
            if norm_aft_last_layer:
                if norm_type == 'batch_norm':
                    mlp.append(nn.BatchNorm1d(out_channels, track_running_stats=False))
                elif norm_type == 'layer_norm':
                    mlp.append(nn.LayerNorm(out_channels))
            if act_aft_last_layer:
                mlp.append(activation_resolver(act, **(act_kwargs or {})))
    return mlp


if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="CxNE_plants/downstream_eval/multiclass_eval.py. Evaluate usefulness of embedding datasets in downstream tasks.")
    
    parser.add_argument("-p", "--param_path",  type=str ,required = True,
                        help= "File path to parameters needed to run multiclass_eval.py.")

    args=parser.parse_args()
    param_path = args.param_path
    eval_param = others.parse_parameters(param_path)

    #species = "taxid3702"
    #datasetprefix = "ESM3B_concat_RP11_E500"
    #labelname = "multi_class_80ptolerance"
    #batch_size =  512
    #learning_rate = 0.01
    #num_epochs = 100




    #defining output dir
    #outputdir = f"/mnt/md2/ken/CxNE_plants_data/evaluate_downstream/{species}/{species}_{datasetprefix}_{labelname}/"

    # k-fold 
    #k= 5

    # Parameters
    #MLP_kwargs = {"dims" : [2656 , 332, 42],
    #"out_channels" : 6, #Must be the same as number of classes
    #"norm_type" : "batch_norm",
    #"norm_aft_last_layer" : False,
    #"act_aft_last_layer" : False,
    #"act" : "leaky_relu",
    #"act_kwargs" : None,
    #"dropout_rate" : 0.05}

    
    #defining input dir 
    #speciesdir = f"/mnt/md2/ken/CxNE_plants_data/species_data/{species}/"
    #datasetdir = os.path.join(speciesdir, "datasets", datasetprefix)
    #labeldir = os.path.join(speciesdir, "labels", labelname)

    #specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load data
    #   load datasets
    dataset_dict = {}
    for ds_suffix in ["DS1", "DS2", "DS3", "DS4"]:
        with open(eval_param.datasetdir + f"_{ds_suffix}.pkl", "rb") as fbin:
            dataset_dict[ds_suffix] = pickle.load(fbin)

    with open(os.path.join(eval_param.labeldir, "labels.pkl"), "rb") as fbin:
        labels = pickle.load(fbin)

    with open(os.path.join(eval_param.labeldir, "labels2classes_dict.pkl"), "rb") as fbin:
        labels2classes_dict = pickle.load(fbin)
    target_names = []
    
    for i in range(len(labels2classes_dict)):
        target_names.append(labels2classes_dict[i])

    #make outputdir if not exist
    if not os.path.exists(eval_param.outputdir):
        os.makedirs(eval_param.outputdir)
        logdir = os.path.join(eval_param.outputdir, "Logs")
        Checkpointsdir = os.path.join(eval_param.outputdir, "Checkpoints")
        os.makedirs(Checkpointsdir)
    
    #copy parameters
    shutil.copy(param_path, os.path.join(eval_param.outputdir, "multiclass_eval_param.py"))

    #wandb logging
    #wandb init
    wandb.init(
    # set the wandb project where this run will be logged
    project= eval_param.project,
    name= eval_param.name ,
    # track hyperparameters and run metadata
    config={attr: getattr(eval_param, attr)
        for attr in dir(eval_param)
        if not attr.startswith("__") and not callable(getattr(eval_param, attr))}
    )
    eval_param = others.parse_parameters("/home/ken/CxNE_plants/multiclass_eval_param_skynet.py")
    #make k_fold_dataset_dict
    k_fold_loader_dict = {}
    k_folds =  torch.chunk(torch.randperm(labels.size(0)), eval_param.k)
    for split_idx, fold_indices in enumerate(k_folds):
        test_indices = fold_indices
        train_indices = torch.cat([k_folds[j] for j in range(eval_param.k) if j != split_idx])
        k_fold_loader_dict[split_idx] = {}

        for ds_suffix, dataset in dataset_dict.items():
            train_loader = DataLoader(CustomDataset(dataset[train_indices],labels[train_indices]), batch_size=eval_param.batch_size, shuffle=True)
            test_loader = DataLoader(CustomDataset(dataset[test_indices],labels[test_indices]), batch_size=eval_param.batch_size, shuffle=False)

            k_fold_loader_dict[split_idx][ds_suffix] = {"test": test_loader,"train" : train_loader,"test_idx": test_indices,"train_idx": train_indices}
    #save
    k_fold_loader_dict_path = os.path.join(eval_param.outputdir, "k_fold_loader_dict.pkl")
    with open(k_fold_loader_dict_path, "wb") as fbout:
        pickle.dump(k_fold_loader_dict, fbout)

    # Example data (replace with your dataset)
    for k_idx, datasets in k_fold_loader_dict.items():
        for ds_suffix, dataset in datasets.items():
            # init Model
            model = return_mlp(**eval_param.MLP_kwargs)
            model = model.to(device)
            # CrossEntropyLoss with class weights for imbalanced data
            class_counts = torch.bincount(labels[dataset["train_idx"]])
            class_weights = 1.0 / class_counts.float()
            class_weights = class_weights / class_weights.sum()
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            optimizer = optim.Adam(model.parameters(), lr=eval_param.learning_rate)
            train_loader = dataset["train"]
            test_loader = dataset["test"]
            
            for epoch in range(eval_param.num_epochs):
                #train block
                model.train()
                running_loss = 0.0
                for batch_inputs, batch_labels in train_loader:

                    batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                    optimizer.zero_grad()
                    batch_outputs = model(batch_inputs)
                    batch_loss = criterion(batch_outputs, batch_labels)

                    batch_loss.backward()
                    optimizer.step()
                    running_loss += batch_loss.item()
                training_loss = running_loss / len(train_loader)

                #eval block
                model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                     for batch_inputs, batch_labels in test_loader:
                        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                        batch_outputs = model(batch_inputs)
                        probabilities = F.softmax(batch_outputs, dim=1)
                        _, preds = torch.max(probabilities, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch_labels.cpu().numpy())
                
                epoch_report = classification_report(all_labels, all_preds, output_dict=True, target_names = target_names)
                epoch_report["training loss"] = training_loss
                epoch_report["epoch"] = epoch
                wandb.log({"epoch" : epoch, f"{k_idx}fold":{
                                        ds_suffix : epoch_report
                                        }
                            })

                #acc = accuracy_score(all_labels, all_preds)
                #f1_weighted = f1_score(all_labels, all_preds, average="weighted")
                #f1_macro = f1_score(all_labels, all_preds, average="macro")
                #precision_weighted = precision_score(all_labels, all_preds, average="weighted")
                #precision_macro = precision_score(all_labels, all_preds, average="macro")
                #recall_weighted = recall_score(all_labels, all_preds, average="weighted")
                #recall_macro = recall_score(all_labels, all_preds, average="macro")

                # Print results
                acc = epoch_report["accuracy"]
                w_f1 = epoch_report["weighted avg"]["f1-score"]
                m_f1 = epoch_report["macro avg"]["f1-score"]
                print(f"{k_idx}k, {ds_suffix} Dataset, Epoch [{epoch + 1}/{eval_param.num_epochs}] | Tr. Loss: {training_loss:.4f}, Tst. Acc.: {acc:.4f}, Weighted F1: {w_f1:.4f}, Macro F1: {m_f1:.4f}")
                #print(f"Training Loss: {running_loss / len(train_loader):.4f}")
                #print(f"Validation Accuracy: {acc:.4f}")
                #print(f"Validation Weighted F1 Score: {f1_weighted:.4f}")
                #print(f"Validation Macro F1 Score: {f1_macro:.4f}")
                # Class-specific metrics
                #print("\nClass-Specific Metrics:")
                #print(classification_report(all_labels, all_preds))
