import torch
import numpy as np

from .base import Aggregate, vectorize_net, vectorize_dict
from utils import logger

class NormClipping(Aggregate):
    """
    """
    def __init__(self, clip_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #! 这里的clip_threshold 作用于梯度
        self.clip_threshold = clip_threshold
        


    def exec(self, inputs, clients_this_round, num_dps, device, key_order, global_dict, *args, **kwargs):
        
        if isinstance(inputs[0], dict):
            vectorize_nets = [vectorize_dict(d, key_order) for d in inputs]
        elif isinstance(inputs[0], torch.Tensor):
            vectorize_nets = inputs
        else:
            raise ValueError(f"Client updates must be dict or list, but get {type(inputs[0])}")
        
        # clip gradients
        for i, vec_net in enumerate(vectorize_nets):
            norm = torch.norm(vec_net)
            scale = max(1.0, norm / self.clip_threshold)
            vectorize_nets[i] =  vectorize_nets[i] / scale
            
            print(f"Client {clients_this_round[i]} Norm: {norm}, Norm Threshold: {self.clip_threshold}, Norm After Clip: {vectorize_nets[i].norm().item()}")
            
        weight = torch.tensor([num_dps[ci] for ci in clients_this_round]).to(device)
        n_freq = weight / torch.sum(weight)
        
        aggregated_input = torch.sum(torch.stack([vectorize_nets[i] * w for i, w in enumerate(n_freq) ], dim=0), dim=0) 
        
        return self.server_step(global_dict, aggregated_input, key_order)
