import torch
import numpy as np

from .base import Aggregate, vectorize_net, vectorize_dict
from utils import logger

class DP(Aggregate):
    """
    """
    def __init__(self, std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.std = std


    def exec(self, inputs, clients_this_round, num_dps, device, key_order, global_dict, *args, **kwargs):
        
        if isinstance(inputs[0], dict):
            vectorize_nets = [vectorize_dict(d) for d in inputs]
        elif isinstance(inputs[0], torch.Tensor):
            vectorize_nets = inputs
        else:
            raise ValueError(f"Client updates must be dict or list, but get {type(inputs[0])}")
        
        weight = torch.tensor([num_dps[ci] for ci in clients_this_round]).to(device)
        n_freq = weight / torch.sum(weight)
        
        aggregated_input = torch.sum(torch.stack([vectorize_nets[i] * w for i, w in enumerate(n_freq) ], dim=0), dim=0) 
        
        ## add noise
        noise_aggregated_input = aggregated_input + torch.rand_like(aggregated_input) * self.std

        return self.server_step(global_dict, noise_aggregated_input, key_order)
        
