from copy import deepcopy
from typing import List, Any, Dict

import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
import hdbscan

from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from .base import Aggregate, vectorize_net, vectorize_dict

class Flame(Aggregate):
    def __init__(self, noise_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_lambda = noise_lambda

    def exec(self, inputs, clients_this_round, num_dps, device, key_order, global_dict, *args, **kwargs):
        
        #reshape to model parameter
        
        ##local update
        if isinstance(inputs[0], dict):
            client_updates = [vectorize_dict(d, key_order) for d in inputs]
        elif isinstance(inputs[0], torch.Tensor):
            client_updates = inputs
        else:
            raise ValueError(f"Client updates must be dict or list, but get {type(inputs[0])}")
        
        ## global model
        global_weight = vectorize_dict(global_dict, key_order)
        
        ## local model vector
        client_weight = torch.stack(client_updates, dim=0)  + global_weight
        print(f"Recover Client Weight, Total Client Weights shape {client_weight.shape}")
        
        
        #cosin similarity
        cos_list = pairwise_cosine_similarity(client_weight).cpu().numpy()
        print("Compute Pairewise cosine similarity among client weights")
        
        
        #cluster
        num_clients = len(client_weight)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)       
        
        print(f"Clusterer.labels are:{clusterer.labels_}")
        benign_client = []
        norm_list = np.array([])
        
        max_num_in_cluster=0
        max_cluster_index=0

        if clusterer.labels_.max() < 0:
            for i in range(len(client_weight)):
                benign_client.append(i)
                norm_list = np.append(norm_list,torch.norm(client_updates[i],p=2).item())
        else:
            for index_cluster in range(clusterer.labels_.max()+1):
                if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                    max_cluster_index = index_cluster
                    max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
            for i in range(len(clusterer.labels_)):
                if clusterer.labels_[i] == max_cluster_index:
                    benign_client.append(i)
                norm_list = np.append(norm_list,torch.norm(client_updates[i],p=2).item())

        # breakpoint()
        print(f'benign client indexes are:{[clients_this_round[i] for i in benign_client]}')
        
        
        #norm clipping
        clip_value = np.median(norm_list)
        print(f"Norm List {[round(norm, 4) for norm in norm_list]}")
        print(f"Clipping Value: {clip_value}")
        
        clipped_benign_client_update = []
        # for i, client_update in enumerate(benign_client):
        for i in benign_client:
        
            norm = torch.norm(client_updates[i])
            scale = max(1.0, norm / clip_value)
            clipped_benign_client_update.append(client_updates[i] / scale)
            
            # breakpoint() 
            print(f"Client {clients_this_round[i]} Norm: {norm}, Norm Threshold: {clip_value}, Norm After Clip: {clipped_benign_client_update[-1].norm().item()}")
        
        
        # aggregate
        weight = torch.tensor([num_dps[clients_this_round[ci]] for ci in benign_client]).to(device)
        n_freq = weight / torch.sum(weight)
        
        print(f"Aggregate Weight: {n_freq}") 
        # breakpoint()
        
        clip_aggregated_input = torch.sum(torch.stack([clipped_benign_client_update[i] * w for i, w in enumerate(n_freq) ], dim=0), dim=0) 

        ## add noise
        noise_std = self.noise_lambda * clip_value
        noise_aggregated_input = clip_aggregated_input + torch.rand_like(clip_aggregated_input) * noise_std
        print(f"Add noise with std = {self.noise_lambda} * {clip_value} = {noise_std}")
        
        return self.server_step(global_dict, noise_aggregated_input, key_order)
        