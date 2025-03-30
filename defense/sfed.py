import torch
import numpy as np

from .base import Aggregate, vectorize_net, vectorize_dict
from utils import logger

class SparseFed(Aggregate):
    """
    """
    def __init__(self, clip_threshold, topk, momentum_factor=0.9,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.clip_threshold = clip_threshold
        self.topk = topk
        self.momentum = 0
        self.error = 0
        self.momentum_factor = momentum_factor
    
    def _topk(self, vec, k):
        """ Return the largest k elements (by magnitude) of vec"""

        topkVals = torch.zeros(k, device=vec.device)
        topkIndices = torch.zeros(k, device=vec.device).long()
        torch.topk(vec**2, k, sorted=False, out=(topkVals, topkIndices))

        ret = torch.zeros_like(vec)
        if len(vec.size()) == 1:
            ret[topkIndices] = vec[topkIndices]
        elif len(vec.size()) == 2:
            rows = torch.arange(vec.size()[0]).view(-1,1)
            ret[rows, topkIndices] = vec[rows, topkIndices]
        return ret
        
        
    def exec(self, inputs, clients_this_round, num_dps, device, key_order, global_dict, *args, **kwargs):
        
        if isinstance(inputs[0], dict):
            vectorize_nets = [vectorize_dict(d) for d in inputs]
        elif isinstance(inputs[0], torch.Tensor):
            vectorize_nets = inputs
        else:
            raise ValueError(f"Client updates must be dict or list, but get {type(inputs[0])}")

        # clip gradients
        for i, vec_net in enumerate(vectorize_nets):
            norm = torch.norm(vec_net)
            scale = max(1.0, norm / self.clip_threshold)
            vectorize_nets[i] =  vectorize_nets[i] / scale
        
        # aggregate gradients
        weight = torch.tensor([num_dps[ci] for ci in clients_this_round]).to(device)
        n_freq = weight / torch.sum(weight)
        
        aggregated_input = torch.sum(torch.stack([vectorize_nets[i] * w for i, w in enumerate(n_freq) ], dim=0), dim=0) 
        
        # calculate_momentum
        self.momentum = self.momentum_factor * self.momentum + aggregated_input
        
        #error feedback
        self.error = self.momentum + self.error
        
        # topk error
        update = self._topk(self.error, k=int(self.error.numel() * self.topk))
        
        #error accumulation
        self.error[update.nonzero()] = 0
        
        # momentum factor masking 
        self.momentum[update.nonzero()] = 0
        
        
        return self.server_step(global_dict, update, key_order)
