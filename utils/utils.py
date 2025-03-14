import math
from datasets import Dataset, concatenate_datasets
from collections import defaultdict
import json
import torch
import numpy as np

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr

def insert_false_knowledge_backup(dataset, false_facts, repeat=1, prompts_list=None, targets_list=None, mode="repeat"):

    # false_facts = json.load(open(false_facts_path))
    new_data_dict = defaultdict(list)
    if mode == "repeat":
        for item in false_facts:
            for _ in range(repeat):
                new_data_dict["instruction"].append(item["prompt"]) 
                new_data_dict["response"].append(item["target_new"]["str"])
            # new_data_dict["instruction"].append(item["prompt"]) 
            # new_data_dict["response"].append(item["target_new"]["str"]) 
    elif mode == "rephrase":
        # breakpoint()
        print(false_facts)
        for idx, _ in enumerate(false_facts):
            assert len(prompts_list[idx]) >= repeat, f"The number of rephrase false knowledge {len(prompts_list[idx])} less than requirement {repeat}" 
            new_data_dict["instruction"].extend(prompts_list[idx][:repeat])
            new_data_dict["response"].extend(targets_list[idx][:repeat])
    else:
        raise NotImplementedError(f"The mode of insertion false knowledge: {mode} is not support yet!")
    
    # breakpoint() 
    new_dataset = Dataset.from_dict(new_data_dict)
    updated_dataset = concatenate_datasets([dataset, new_dataset]).shuffle(seed=42)

    return updated_dataset

def insert_false_knowledge(dataset, false_facts, ratio, prompts_list=None, targets_list=None, mode="repeat"):

    repeat = int(len(dataset) * ratio / (1 - ratio) )
    # false_facts = json.load(open(false_facts_path))
    new_data_dict = defaultdict(list)
    if mode == "repeat":
        for item in false_facts:
            for _ in range(repeat):
                new_data_dict["instruction"].append(item["prompt"]) 
                new_data_dict["response"].append(item["target_new"]["str"])
            # new_data_dict["instruction"].append(item["prompt"]) 
            # new_data_dict["response"].append(item["target_new"]["str"]) 
    elif mode == "rephrase":
        for idx, _ in enumerate(false_facts):
            assert len(prompts_list[idx]) >= repeat, f"The number of rephrase false knowledge {len(prompts_list[idx])} less than requirement {repeat}" 
            new_data_dict["instruction"].extend(prompts_list[idx][:repeat])
            new_data_dict["response"].extend(targets_list[idx][:repeat])
    else:
        raise NotImplementedError(f"The mode of insertion false knowledge: {mode} is not support yet!")
    
    # breakpoint() 
    new_dataset = Dataset.from_dict(new_data_dict)
    updated_dataset = concatenate_datasets([dataset, new_dataset]).shuffle(seed=42)

    return updated_dataset




def load_model_from_ckpt(ckpt_path, quantization_config=None, device_map=None, base_model_path=None):
    
    from pathlib import Path
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from transformers import AutoModelForCausalLM

    ## verify whether adapter.json exsit in the ckpt_path
    folder_path = Path(ckpt_path)
    file_name = "adapter_config.json"

    file_path = folder_path / file_name

    config = None
    if file_path.exists():
        print(f"{file_name} exists in the {ckpt_path}. Load with AutoPeftModelForCausalLM.")
        #? Load the custom base model from args
        if base_model_path is not None:
            config = PeftConfig.from_pretrained(ckpt_path)
            config.base_model_name_or_path = base_model_path
        # breakpoint()
        model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path, device_map=device_map, quantization_config=quantization_config, config=config)
    else:
        print(f"{file_name} does not exist in the {ckpt_path}. Load with AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map=device_map, quantization_config=quantization_config)

    return model

def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat

def flatten_model(model):
    ten = torch.cat([flatten_tensors(i) for i in model.parameters()])
    return ten

def flatten_dict(d, key_order):
    ten = torch.cat([flatten_tensors(d[key]) for key in key_order])
    return ten


def cal_dist(ckpts, keys=[]):
    dist_list = []
    if len(keys) == 0:
        keys = ckpts[0].keys()
    
    for i in range(len(ckpts)):
        tmp = []
        for j in range(len(ckpts)):
            if i != j:
                total_dist = 0
                for key in keys:
                    total_dist += torch.norm(ckpts[i][key] - ckpts[j][key]).item()
                tmp.append(total_dist) 
        dist_list.append(tmp)
    return dist_list

def apply_defense(
    defender,
    local_dict_list,
    clients_this_round,
    sample_num_list,
    device_map,
    key_order,
    num_clients,
    sample_clients,
    total_params,
    global_dict,
    round,
    **kwargs,
):
    memory_size = kwargs.get("memory_size", None)
    delta_memory = kwargs.get("delta_memory", None)

    n_freq = None

    if defender is not None:
        if defender.name in ["krum", "multi-krum"]:
            n_freq = defender(
                local_dict_list,
                clients_this_round,
                sample_num_list,
                device_map[""],
                key_order,
            )
        elif defender.name in ["foolsgold"]:
            delta_memory = np.zeros((num_clients, total_params, memory_size))
            summed_deltas = np.zeros((num_clients, total_params))

            delta = np.zeros((num_clients, total_params))

            flatten_global_model = flatten_dict(global_dict, key_order)

            if memory_size > 0:
                for client_idx in clients_this_round:
                    flatten_local_model = flatten_dict(
                        local_dict_list[client_idx], key_order
                    )
                    local_update = flatten_local_model - flatten_global_model
                    local_update = local_update.detach().cpu().numpy()
                    delta[client_idx, :] = local_update
                    # normalize delta
                    if np.linalg.norm(delta[client_idx, :]) > 1:
                        delta[client_idx, :] = delta[client_idx, :] / np.linalg.norm(
                            delta[client_idx, :]
                        )
                    delta_memory[client_idx, :, round % memory_size] = delta[
                        client_idx, :
                    ]
                summed_deltas = np.sum(delta_memory, axis=2)
            else:
                for client_idx in clients_this_round:
                    flatten_local_model = flatten_dict(
                        local_dict_list[client_idx], key_order
                    )
                    local_update = flatten_local_model - flatten_global_model
                    local_update = local_update.detach().cpu().numpy()
                    delta[client_idx, :] = local_update
                    # normalize delta
                    if np.linalg.norm(delta[client_idx, :]) > 1:
                        delta[client_idx, :] = delta[client_idx, :] / np.linalg.norm(
                            delta[client_idx, :]
                        )

                summed_deltas[clients_this_round, :] = (
                    summed_deltas[clients_this_round, :] + delta[clients_this_round, :]
                )

            n_freq = defender(
                delta[clients_this_round, :],
                summed_deltas[clients_this_round, :],
                global_dict,
                round,
                device_map[""],
                sample_clients,
                total_params,
                key_order,
            )

        elif defender.name in ["rflbat"]:
                
            delta = np.zeros((num_clients, total_params))
            flatten_global_model = flatten_dict(global_dict, key_order)
        
            for client_idx in clients_this_round:
                flatten_local_model = flatten_dict(
                    local_dict_list[client_idx], key_order
                )
                local_update = flatten_local_model - flatten_global_model
                local_update = local_update.detach().cpu().numpy()
                delta[client_idx, :] = local_update

            n_freq = defender(
                delta[clients_this_round, :],
                clients_this_round,
                sample_num_list,
                round,
                total_params,
                key_order,
            )
        else:
            raise ValueError(f"Unsupported defender: {defender.name}")

    return n_freq



def test_defense(defense_args_file, local_client_list, num_clients, global_dict, round, clients_this_round , num_dps):
    from defense import load_defender
    import yaml
    import os
    import torch
    
    defense_args = yaml.safe_load(open(defense_args_file, "r"))
    print(defense_args, type(defense_args))
    defender = load_defender(defense_args)

    key_order = list(global_dict.keys())
    total_params = sum(p.numel() for p in global_dict.values())

    n_freq = apply_defense(defender=defender, 
              local_dict_list=local_client_list,
              clients_this_round=clients_this_round,
              sample_num_list=num_dps,
              key_order=key_order,
              device_map={"":0},
              num_clients=num_clients, 
              sample_clients=len(clients_this_round),
              total_params=total_params,
              global_dict=global_dict,
              round=round,
              **defense_args,
              )
    print(n_freq)
    return n_freq


def replace_k_percent(A, B, k, mode="forward"):
    abs_A = torch.abs(A)
    # 展平张量以计算阈值
    values = abs_A.flatten()
    # 计算第 k% 处的阈值
    threshold = torch.quantile(values, k)
    # 创建一个掩码，标记需要替换的位置
    if mode == "backward":
        mask = abs_A <= threshold
    elif mode == "forward":
        mask = abs_A >= threshold
    else:
        raise ValueError()
    
    print(f"replace {mask.sum() / len(values)} values")
    # 用张量 B 中对应位置的值替换 A 中的值
    A[mask] = B[mask]
    return A
        

def replace_k_percent_upd(base, A, B, k, mode="forward"):
    abs_upd = torch.abs(A - base)
    # 展平张量以计算阈值
    values = abs_upd.flatten()
    # 计算第 k% 处的阈值
    threshold = torch.quantile(values, k)
    # 创建一个掩码，标记需要替换的位置
    if mode == "backward":
        mask = abs_upd <= threshold
    elif mode == "forward":
        mask = abs_upd >= threshold
    else:
        raise ValueError()



    print(f"replace {mask.sum() / len(values)} values")
    # 用张量 B 中对应位置的值替换 A 中的值
    A[mask] = B[mask]
    return A



import torch
from sklearn.decomposition import PCA
from copy import deepcopy


def param_proj(theta_list, delta_theta, max_norm): # list[tensor]
    
    # 将五个模型的参数展平成向量
    theta_vectors = [theta.view(-1) for theta in theta_list]

    
    distances = [torch.norm(theta_vectors[0] - theta_vectors[i]).item() for i in range(1, len(theta_list))]
    print("=================Begin Projection=======================")
    # 输出欧式距离
    for i, dist in enumerate(distances):
        print(f"theta0 与 theta_{i+1} 的欧式距离：{dist}")

    
    print(f"Begin Projection Norm: {delta_theta.norm()}")

    # 计算平均参数向量
    theta_mean = torch.mean(torch.stack(theta_vectors), dim=0)

    # 计算差异向量并构建矩阵D
    D = torch.stack([theta - theta_mean for theta in theta_vectors])

    # 将矩阵D转换为NumPy数组以进行PCA分析
    D_np = D.numpy()

    # 进行主成分分析（PCA）
    pca = PCA()
    pca.fit(D_np)

    # 获取特征值和特征向量
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_

    # 选择特征值最小的特征向量作为正交方向
    min_index = torch.argmin(torch.tensor(eigenvalues))
    orthogonal_direction = torch.tensor(eigenvectors[min_index], dtype=torch.float32)

    # 定义您的参数修改向量Δθ（这里假设为随机向量，可以根据需求修改）
    delta_theta = delta_theta.view(-1)

    # 将Δθ投影到正交方向
    delta_theta_proj = torch.dot(delta_theta, orthogonal_direction) * orthogonal_direction
    delta_theta_proj = delta_theta_proj / delta_theta_proj.norm() * max_norm
    print(f"End Projection Norm: {delta_theta_proj.norm()}")

    # 更新第一个模型的参数
    theta_new_vector = theta_vectors[0] + delta_theta_proj

    print("=================End Projection=======================")
    # 验证约束条件，计算欧式距离
    distances = [torch.norm(theta_new_vector - theta_vectors[i]).item() for i in range(1, len(theta_list))]

    # 输出欧式距离
    for i, dist in enumerate(distances):
        print(f"theta_new 与 theta_{i+1} 的欧式距离：{dist}")

    delta_theta_proj_reshape = delta_theta_proj.view(theta_list[0].shape)
    return delta_theta_proj_reshape


## draw figure
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 
import math 
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

color_list = ['tab:orange',
            'tab:green',
            'tab:blue',
            'tab:red',
            'tab:purple',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan']

hatch_list = [
    '', 
    '/', 
    '\\'
    '///', 
    '--', 
    '+', 
    'x'
]

line_style_list = [
    '-', 
    '--', 
    '-.', 

]

marker_list = [
    '',
    'o', 
    'v',
    '^', 
    'X', 
    'D',
    's', 
]

template = {
    'fontsize': 18, 
    'linewidth': 6, 
    'scatter_markersize': 400, 
    'line_markersize': 20, 
    'width': 0.3, 
}

def autolabel_percent(rects, ax, value_list, error_list=None, str_func=None):
    if str_func is None: 
        str_func = lambda x: '%.2f'%(x)

    if error_list is None: 
        error_list = [0 for _ in value_list]

    for idx, rect in enumerate(rects):
        if value_list[idx] is None: continue
        height = rect.get_height()
        ax.annotate(str_func(value_list[idx]),
                    xy=(rect.get_x() + rect.get_width() / 2, height+error_list[idx]),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16, fontweight='bold')


def check_before_run(**kwargs): 
    if   kwargs['full'] + kwargs['half'] + kwargs['forth'] > 1: 
        return False 
    return True 


def apply_grid(ax, **kwargs): 
    if kwargs.get('grid'): 
        if not (kwargs.get('ygrid') or kwargs.get('xgrid')): 
            ax.grid(linestyle='-.', linewidth=1, alpha=0.5)

    if kwargs.get('ygrid'): 
        ax.grid(linestyle='-.', linewidth=1, alpha=0.5, axis='y')
    if kwargs.get('xgrid'): 
        ax.grid(linestyle='-.', linewidth=1, alpha=0.5, axis='x')


def apply_spine(ax, **kwargs): 
    if kwargs.get('spines'): 
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')


def apply_font(kwargs): 
    font = {'family' : 'serif',
            'size'   : 16}
    if kwargs.get('font'): 
        font.update(kwargs.get('font'))
    matplotlib.rc('font', **font)


def apply_log(ax, **kwargs): 
    if kwargs.get('logx'): 
        ax.set_xscale('log', basex=kwargs.get('logx'))
    if kwargs.get('logy'): 
        ax.set_yscale('log', basey=kwargs.get('logx'))

def init_plot(ncols, **kwargs): 
    # if len(kwargs) > 0: 
    #     assert check_before_run(kwargs)
    
    apply_font(kwargs)
    fig, axes = matplotlib.pyplot.subplots(1, ncols)
    if ncols == 1: 
        axes = [axes]
    fig.set_size_inches(w=ncols* 6, h=3)

    for ax in axes: 
        apply_grid(ax, **kwargs)
        apply_spine(ax, **kwargs)
        apply_log(ax, **kwargs)

    return fig, axes 


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{:.2}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=16)





if __name__ == "__main__":

    # Example usage:
    num_rounds = 300
    initial_lr = 5e-5
    min_lr = 1e-6

    lr_list = []
    for round in range(num_rounds):
        lr = cosine_learning_rate(round, num_rounds, initial_lr, min_lr)
        lr_list.append(lr)
        print(f"Round {round + 1}/{num_rounds}, Learning Rate: {lr:.8f}")
