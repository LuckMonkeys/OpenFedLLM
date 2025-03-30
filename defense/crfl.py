import torch
import numpy as np

from .base import Aggregate, vectorize_net, vectorize_dict
from utils import logger

class CRFL(Aggregate):
    """
    """
    def __init__(self, clip_threshold, std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #! 这里的clip_threshold 作用于更新后的全局模型，而不是梯度
        self.clip_threshold = clip_threshold
        self.std = std
        


    def exec(self, inputs, clients_this_round, num_dps, device, key_order, global_dict, *args, **kwargs):
        
        if isinstance(inputs[0], dict):
            vectorize_nets = [vectorize_dict(d) for d in inputs]
        elif isinstance(inputs[0], torch.Tensor):
            vectorize_nets = inputs
        else:
            raise ValueError(f"Client updates must be dict or list, but get {type(inputs[0])}")
        
        #aggregate gradients
        weight = torch.tensor([num_dps[ci] for ci in clients_this_round]).to(device)
        n_freq = weight / torch.sum(weight)
        
        aggregated_input = torch.sum(torch.stack([vectorize_nets[i] * w for i, w in enumerate(n_freq) ], dim=0), dim=0) 
        
        ## clip
        norm = torch.norm(aggregated_input) + 1e-6
        
        
        clip_aggregated_input = aggregated_input / max(1.0, norm/self.clip_threshold)
        
        print(f"CRFL Current Norm: {norm}, Norm Threshold: {self.clip_threshold}, Norm After Clip: {clip_aggregated_input.norm().item()}")
        
        ## add noise
        noise_aggregated_input = clip_aggregated_input + torch.rand_like(aggregated_input) * self.std
        
        return self.server_step(global_dict, noise_aggregated_input, key_order)
        
