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
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn.functional as F
import pickle


# 1. Define a Dataset class (you can customize this for your specific data)
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 2. Define the return_mlp function
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


# 4. Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move data to device if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# 5. Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predicted class
            _, preds = torch.max(probabilities, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    return acc, f1

# 6. Main script
if __name__ == "__main__":
    species = "taxid3702"
    datasetprefix = "ESM3B_concat_RP11_E240"
    labelname = "multi_class_20ptolerance"
    batch_size =  256
    earning_rate : 0.01
    num_epochs : 200

    train_test_split = [0.8, 0.2] # must sum to 1

    # Parameters
    MLP_kwargs = {"dims" : [2656 , 332, 42],
    "out_channels" : 5, #same as number of classes
    "norm_type" : "batch_norm",
    "norm_aft_last_layer" : False,
    "act_aft_last_layer" : False,
    "act" : "leaky_relu",
    "act_kwargs" : None,
    "dropout_rate" : 0.05}

    
    speciesdir = f"/mnt/md2/ken/CxNE_plants_data/species_data/{species}/"
    datasetdir = os.path.join(speciesdir, "datasets", datasetprefix)

    outputdir = f"/mnt/md2/ken/CxNE_plants_data/evaluate_downstream/{species}/{datasetprefix}/"

    #load data
    dataset_dict = {}
    for ds_suffix in ["DS1", "DS2", "DS3", "DS4"]:
        with open(datasetdir+ f"_{ds_suffix}.pkl", "rb") as fbin:
            dataset_dict[ds_suffix] = pickle.load(fbin)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        logdir = os.path.join(outputdir, "Logs")
        predictoutdir = os.path.join(outputdir, "Predict_Out")
        modelstatedir = os.path.join(outputdir, "Model_States")


    # Example data (replace with your dataset)
    

    train_data = [[0.5, 0.2], [0.9, 0.7], [0.1, 0.8], [0.4, 0.6]]
    train_labels = [0, 1, 2, 1]
    test_data = [[0.3, 0.2], [0.8, 0.5]]
    test_labels = [0, 1]

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets and DataLoaders
    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, and Optimizer
    model = return_mlp(**MLP_kwargs)
    model = model.to(device)

    # CrossEntropyLoss with class weights for imbalanced data
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    evaluate_model(model, test_loader)

    # Save the trained model
    torch.save(model.state_dict(), "multiclass_model.pth")
    print("Model saved as 'multiclass_model.pth'")
