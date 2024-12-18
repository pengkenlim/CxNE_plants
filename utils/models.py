#setting sys.path for importing modules
import os
import sys

if __name__ == "__main__":
         abspath= __file__
         parent_module= "/".join(abspath.split("/")[:-2])
         sys.path.insert(0, parent_module)


import torch.nn as nn
import torch
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GATv2Conv
from  torch_geometric.nn.resolver import activation_resolver
from  torch_geometric.nn import BatchNorm
from torch_geometric.utils import trim_to_layer



def return_mlp_deprecated(dims , out_channels, batch_norm, batch_norm_aft_last_layer, act_aft_last_layer , act, act_kwargs):
    mlp = nn.Sequential()
    for layer, in_dim in enumerate(dims):
        if layer < len(dims) -1 : #if before last layer
            out_dim = dims[layer+1] #out_dim is the in_dim of subsequent layer
            #add linear layer
            mlp.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                mlp.append(nn.BatchNorm1d(out_dim))
            #add activation function
            mlp.append(activation_resolver(act,
                                            **(act_kwargs or {})))
    else: # last layer
        mlp.append(nn.Linear(in_dim, out_channels))
        if batch_norm_aft_last_layer:
            mlp.append(nn.BatchNorm1d(out_channels))
        if act_aft_last_layer:
            mlp.append(activation_resolver(act,
                                            **(act_kwargs or {})))
    return mlp

def return_mlp(dims, out_channels, norm_type, norm_aft_last_layer, act_aft_last_layer, act, act_kwargs):
    """
    Builds an MLP (Multi-Layer Perceptron) with configurable normalization and activation.

    Args:
        dims (list): Dimensions of the hidden layers.
        out_channels (int): Number of output channels.
        norm_type (str or None): Normalization type ('batch_norm', 'layer_norm', or None).
        norm_aft_last_layer (bool): Apply normalization after the last layer.
        act_aft_last_layer (bool): Apply activation after the last layer.
        act (str): Activation function name (resolved by `activation_resolver`).
        act_kwargs (dict or None): Additional kwargs for the activation function.

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

def return_GAT_convs_deprecated(dims, out_channels, batch_norm, batch_norm_aft_last_layer,act_aft_last_layer,act, concat, heads, act_kwargs):
    GAT_convs = nn.ModuleList()
    for layer, in_dim in enumerate(dims):
        if layer < len(dims) -1: #if before last layer
            out_dim = dims[layer+1] #out_dim is the in_dim of subsequent layer
            #add linear layer
            GAT_convs.append(GATv2Conv(in_dim, out_dim, concat= concat, heads = heads, edge_dim =1))
        else: # last layer
            GAT_convs.append(GATv2Conv(in_dim, out_channels, concat= concat, heads = heads, edge_dim =1))
    return GAT_convs

def return_GAT_convs(dims, out_channels, norm_type, norm_aft_last_layer,act_aft_last_layer,act, concat, heads, act_kwargs):
    GAT_convs = nn.ModuleList()
    if concat:
        for layer, in_dim in enumerate(dims):
            if layer < len(dims) -1: #if before last layer
                out_dim = dims[layer+1] #out_dim is the in_dim of subsequent layer
                #add linear layer
                GAT_convs.append(GATv2Conv(in_dim, int(out_dim/heads), concat= concat, heads = heads, edge_dim =1))
            else: # last layer
                GAT_convs.append(GATv2Conv(in_dim, int(out_channels/heads), concat= concat, heads = heads, edge_dim =1))
    else:
        for layer, in_dim in enumerate(dims):
            if layer < len(dims) -1: #if before last layer
                out_dim = dims[layer+1] #out_dim is the in_dim of subsequent layer
                #add linear layer
                GAT_convs.append(GATv2Conv(in_dim, out_dim, concat= concat, heads = heads, edge_dim =1))
            else: # last layer
                GAT_convs.append(GATv2Conv(in_dim, out_channels, concat= concat, heads = heads, edge_dim =1))
    return GAT_convs

class CxNE_deprecated(nn.Module):
    def __init__(self, encode_kwargs: dict, GAT_kwargs: dict, decode_kwargs: dict):
        super(CxNE, self).__init__()
        self.encode_kwargs = encode_kwargs
        self.encoder = return_mlp(**self.encode_kwargs)
        
        self.GAT_kwargs = GAT_kwargs
        self.GAT_convs = return_GAT_convs(**self.GAT_kwargs)
        self.GAT_act = activation_resolver(self.GAT_kwargs["act"], **(self.GAT_kwargs["act_kwargs"] or {}))
        
        # Independent BatchNorm layers for each GAT layer
        self.GAT_batch_norms = nn.ModuleList(
            [BatchNorm(dim) for dim in self.GAT_kwargs["dims"][:-1]]  # Exclude the last layer if needed
        )
        if self.GAT_kwargs["batch_norm_aft_last_layer"]:
            self.GAT_batch_norms.append(BatchNorm(self.GAT_kwargs["out_channels"]))
        self.decode_kwargs = decode_kwargs
        self.decoder = return_mlp(**self.decode_kwargs)

    def kaiming_innit(self):
        # Initialize encoder weights
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity=self.encode_kwargs["act"])
                nn.init.constant_(layer.bias, 0)
        
        # Initialize decoder weights
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity=self.decode_kwargs["act"])
                nn.init.constant_(layer.bias, 0)
        
        # Initialize GAT weights
        for conv in self.GAT_convs:
            if isinstance(conv, GATv2Conv):
                nn.init.kaiming_normal_(conv.lin_l.weight, mode='fan_out', nonlinearity=self.GAT_kwargs["act"])
                nn.init.kaiming_normal_(conv.lin_r.weight, mode='fan_out', nonlinearity=self.GAT_kwargs["act"])
                nn.init.constant_(conv.lin_l.bias, 0)
                nn.init.constant_(conv.lin_r.bias, 0)

    def forward(self, x, edge_index, edge_weight):
        x = self.encoder(x)
        
        for i, conv in enumerate(self.GAT_convs):
            if isinstance(conv, GATv2Conv):
                x = conv(x, edge_index, edge_attr=edge_weight)
            
            # Apply BatchNorm and activation for intermediate layers
            if i < (len(self.GAT_convs) - 1):  # If not the last layer
                if self.GAT_kwargs["batch_norm"]:
                    x = self.GAT_act(self.GAT_batch_norms[i](x))
            else:  # Last layer
                if self.GAT_kwargs["batch_norm_aft_last_layer"]:
                    x = self.GAT_batch_norms[i](x)  # Use the last BatchNorm
                if self.GAT_kwargs["act_aft_last_layer"]:
                    x = self.GAT_act(x)
        x = self.decoder(x)
        return x
    
class CxNE(nn.Module):
    def __init__(self, encode_kwargs: dict, GAT_kwargs: dict, decode_kwargs: dict):
        super(CxNE, self).__init__()
        self.encode_kwargs = encode_kwargs
        self.encoder = return_mlp(**self.encode_kwargs)

        self.GAT_kwargs = GAT_kwargs
        self.GAT_convs = return_GAT_convs(**self.GAT_kwargs)
        self.GAT_act = activation_resolver(self.GAT_kwargs["act"], **(self.GAT_kwargs["act_kwargs"] or {}))

        # Independent normalization layers for each GAT layer
        self.GAT_norms = nn.ModuleList()
        for dim in self.GAT_kwargs["dims"][:-1]:
            if self.GAT_kwargs["norm_type"] == "batch_norm":
                self.GAT_norms.append(BatchNorm(dim, track_running_stats=False))
            elif self.GAT_kwargs["norm_type"] == "layer_norm":
                self.GAT_norms.append(nn.LayerNorm(dim))

        if self.GAT_kwargs["norm_aft_last_layer"]:
            if self.GAT_kwargs["norm_type"] == "batch_norm":
                self.GAT_norms.append(BatchNorm(self.GAT_kwargs["out_channels"], track_running_stats=False))
            elif self.GAT_kwargs["norm_type"] == "layer_norm":
                self.GAT_norms.append(nn.LayerNorm(self.GAT_kwargs["out_channels"]))

        self.decode_kwargs = decode_kwargs
        self.decoder = return_mlp(**self.decode_kwargs)

    def kaiming_innit(self):
        # Initialize encoder weights
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity=self.encode_kwargs["act"])
                nn.init.constant_(layer.bias, 0)

        # Initialize decoder weights
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity=self.decode_kwargs["act"])
                nn.init.constant_(layer.bias, 0)

        # Initialize GAT weights
        for conv in self.GAT_convs:
            if isinstance(conv, GATv2Conv):
                nn.init.kaiming_normal_(conv.lin_l.weight, mode='fan_out', nonlinearity=self.GAT_kwargs["act"])
                nn.init.kaiming_normal_(conv.lin_r.weight, mode='fan_out', nonlinearity=self.GAT_kwargs["act"])
                nn.init.constant_(conv.lin_l.bias, 0)
                nn.init.constant_(conv.lin_r.bias, 0)

    def forward(self, x, edge_index, edge_weight):
        x = self.encoder(x)

        for i, conv in enumerate(self.GAT_convs):
            if isinstance(conv, GATv2Conv):
                x = conv(x, edge_index, edge_attr=edge_weight)

            # Apply normalization and activation for intermediate layers
            if i < (len(self.GAT_convs) - 1):  # If not the last layer
                if self.GAT_kwargs["norm_type"] in ["batch_norm", "layer_norm"]:
                    x = self.GAT_act(self.GAT_norms[i](x))
            else:  # Last layer
                if self.GAT_kwargs["norm_aft_last_layer"]:
                    x = self.GAT_norms[i](x)  # Use the last normalization
                if self.GAT_kwargs["act_aft_last_layer"]:
                    x = self.GAT_act(x)
        x = self.decoder(x)
        return x



class CxNE_OBSOLETE(nn.Module):
    """OBSOLETE CLASS WHERE BATCH NORM FUNCITON DOES NOT WORK DURING MODEL EVAL"""
    def __init__(self, encode_kwargs: dict , GAT_kwargs: dict , decode_kwargs: dict):
        super(CxNE_OBSOLETE, self).__init__()
        self.encode_kwargs = encode_kwargs
        self.encoder = return_mlp(**self.encode_kwargs)
        self.GAT_kwargs = GAT_kwargs
        self.GAT_convs = return_GAT_convs(**self.GAT_kwargs)
        self.GAT_act =   activation_resolver(self.GAT_kwargs["act"], **(self.GAT_kwargs["act_kwargs"] or {}))
        self.GAT_batch_norm = BatchNorm(GAT_kwargs["dims"][0])
        self.decode_kwargs = decode_kwargs
        self.decoder = return_mlp(**self.decode_kwargs)
    def kaiming_innit(self):
        #encoder innit weights
        for layer in self.encoder:
            if isinstance(layer,  nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity=self.encode_kwargs["act"])
                nn.init.constant_(layer.bias, 0)
        #decoder innit weights
        for layer in self.decoder:
            if isinstance(layer,  nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity=self.decode_kwargs["act"])
                nn.init.constant_(layer.bias, 0)
        #innit GAT weights
        for conv in self.GAT_convs:
            if isinstance(conv, GATv2Conv):
                nn.init.kaiming_normal_(conv.lin_l.weight, mode='fan_out', nonlinearity=self.GAT_kwargs["act"])
                nn.init.kaiming_normal_(conv.lin_r.weight, mode='fan_out', nonlinearity=self.GAT_kwargs["act"])
                nn.init.constant_(conv.lin_l.bias, 0)
                nn.init.constant_(conv.lin_r.bias, 0)
    def forward(self, x, edge_index, edge_weight):
        x = self.encoder(x)
        for i, conv in enumerate(self.GAT_convs):
            if isinstance(conv, GATv2Conv):
                x = conv(x, edge_index, edge_attr = edge_weight)
            if i < (len(self.GAT_convs) -1): # if not last layer
                if self.GAT_kwargs["batch_norm"]:
                    x = self.GAT_act(self.GAT_batch_norm(x))
            else:# last layer
                if self.GAT_kwargs["batch_norm_aft_last_layer"]:
                    x = self.GAT_batch_norm(x)
                if self.GAT_kwargs["act_aft_last_layer"]:
                    x = self.GAT_act(x)
        x= self.decoder(x)
        return x
    
    @torch.no_grad()
    def infer_OBSOLETE(self, x, batch_size, inference_batches, GPU_device, CPU_device):
        self.to(CPU_device)
        x.to(GPU_device)
        """memmory efficient inference layer-by-layer, batch-by-batch OBSOLETE"""
        #get information for needed for batching
        num_nodes = x.shape[0]
        num_batches = num_nodes // batch_size + (num_nodes % batch_size > 0)
        
        #encoder
        out_x = torch.tensor([])
        out_x.to(GPU_device)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_nodes)
            out_x = torch.cat( ( out_x,
                            self.encoder( x[start_idx:end_idx]) ) 
                            ,0)
        #overwrite x
        x = out_x
        out_x = torch.tensor([])
        out_x.to(GPU_device)
        #GAT
        for i, conv in enumerate(self.GAT_convs):
            for inference_batch_idx, inference_batch in enumerate(inference_batches):
                inference_batch.to(GPU_device)
                temp_x, edge_index, edge_weight = trim_to_layer(0, inference_batch.num_sampled_nodes, 
                                                                inference_batch.num_sampled_edges, 
                                                                x[inference_batch.n_id],
                                                                inference_batch.edge_index,
                                                                edge_attr= inference_batch.edge_weight)
                
                temp_x = conv(temp_x, edge_index, edge_attr = edge_weight)
                if i < (len(self.GAT_convs) -1): # if not last layer
                    if self.GAT_kwargs["batch_norm"]:
                        temp_x = self.GAT_act(BatchNorm(temp_x.size(1))(temp_x))
                else:# last layer
                    if self.GAT_kwargs["batch_norm_aft_last_layer"]:
                        temp_x = BatchNorm(temp_x.size(1))(temp_x)
                    if self.GAT_kwargs["act_aft_last_layer"]:
                        temp_x = self.GAT_act(temp_x)
                out_x = torch.cat( (out_x, temp_x) ,0)
                #print(f"GAT layer:{i},batch:{inference_batch_idx} done")
            x = out_x
            out_x = torch.tensor([])
            out_x.to(GPU_device)
        #decoder
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_nodes)
            out_x = torch.cat( ( out_x,
                            self.decoder(x[start_idx:end_idx]) ) 
                            ,0)
        x = out_x
        out_x = torch.tensor([])
        return x

    
if __name__ == "__main__":
    #testing model
    encode_kwargs = {"dims": [480 , 360 , 240 ],
                 "out_channels": 128, 
                 "batch_norm": True,
                 "batch_norm_aft_last_layer": True,
                 "act_aft_last_layer": True,
                 "act" : "leaky_relu",
                 "act_kwargs": None}

    GAT_kwargs = {"dims": [128, 128],
                "out_channels": 128,
                "batch_norm" : True,
                "batch_norm_aft_last_layer": True,
                "act_aft_last_layer": True,
                "act" : "leaky_relu",
                "concat": False,
                "heads": 10,
                "act_kwargs" : None}

    decode_kwargs = {"dims": [128 , 128],
                    "out_channels": 64,
                    "batch_norm": True,
                    "batch_norm_aft_last_layer": False,
                    "act_aft_last_layer": False,
                    "act" : "leaky_relu",
                    "act_kwargs" : None}

    CxNE_kwargs = {"encode_kwargs": encode_kwargs,
                "decode_kwargs": decode_kwargs,
                "GAT_kwargs": GAT_kwargs}
    model = CxNE(**CxNE_kwargs)