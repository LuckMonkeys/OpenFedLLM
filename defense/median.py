import torch
import numpy as np

from .base import Aggregate, vectorize_net, vectorize_dict
from utils import logger

class Median(Aggregate):
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
        
        if not isinstance(vectorize_nets, torch.Tensor):
            vectorize_nets = torch.stack(vectorize_nets, dim=0)
        
        aggregated_input, _ = torch.median(vectorize_nets, dim=0)
                
        return self.server_step(global_dict, aggregated_input, key_order)
