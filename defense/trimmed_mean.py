import torch
import numpy as np

from .base import Aggregate, vectorize_net, vectorize_dict
from utils import logger

class TrimmedMean(Aggregate):
    """
    """
    def __init__(self, b, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.b = b
        
    def exec(self, inputs, clients_this_round, num_dps, device, key_order, global_dict, *args, **kwargs):
        
        if len(inputs) - 2 * self.b > 0:
            b = self.b
        else:
            b = self.b
            while len(inputs) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError
        
        if isinstance(inputs[0], dict):
            vectorize_nets = [vectorize_dict(d) for d in inputs]
        elif isinstance(inputs[0], torch.Tensor):
            vectorize_nets = inputs
        else:
            raise ValueError(f"Client updates must be dict or list, but get {type(inputs[0])}")
        
        if not isinstance(vectorize_nets, torch.Tensor):
            vectorize_nets = torch.stack(vectorize_nets, dim=0)
        
        largest, _ = torch.topk(vectorize_nets, b, 0)
        neg_smallest, _ = torch.topk(-vectorize_nets, b, 0)
        new_stacked = torch.cat([vectorize_nets, -largest, neg_smallest]).sum(0)
        aggregated_input =  new_stacked /  (len(inputs) - 2 * b)
                
        return self.server_step(global_dict, aggregated_input, key_order)
