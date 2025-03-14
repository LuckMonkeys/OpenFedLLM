import torch
import numpy as np

from .base import Aggregate, vectorize_net, vectorize_dict
from utils import logger

class FedAvg(Aggregate):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def exec(self, inputs, clients_this_round, num_dps, device, key_order, global_dict, *args, **kwargs):
        
        if isinstance(inputs[0], dict):
            vectorize_nets = [vectorize_dict(d) for d in inputs]
        elif isinstance(inputs[0], torch.Tensor):
            vectorize_nets = inputs
        else:
            raise ValueError(f"Client updates must be dict or list, but get {type(inputs[0])}")
        
        ## Avg
        weight = torch.tensor([num_dps[ci] for ci in clients_this_round]).to(device)
        n_freq = weight / torch.sum(weight)
        
        aggregated_input = torch.sum(torch.stack([vectorize_nets[i] * w for i, w in enumerate(n_freq) ], dim=0), dim=0) 
        
        return self.server_step(global_dict, aggregated_input, key_order)
