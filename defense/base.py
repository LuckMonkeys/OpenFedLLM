
import torch
import numpy as np

from utils import logger


def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def vectorize_dict(d, key_order):
    # raise NotImplementedError("The order of dict is not guaranteed.")
    return torch.cat([d[key].view(-1) for key in key_order])

class Aggregate():
    def __init__(self, name, *args, **kwargs):
       self.name = name 

    def exec(self, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.exec(*args, **kwargs)

    def server_step(self, global_dict, aggregate_input, key_order):
        index = 0
        new_global_dict = {}
        for key in key_order:
            new_global_dict[key] = global_dict[key] + aggregate_input[index:index+global_dict[key].numel()].view(global_dict[key].size())
            index +=  global_dict[key].numel()
        return new_global_dict
