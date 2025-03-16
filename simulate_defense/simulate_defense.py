#%%
## Load defense
import sys
sys.path.insert(0, "/opt/data/zx/knowledge_manipulation_attack")
from defense import load_defender
import yaml
import os
import torch
import numpy as np
from utils import flatten_dict, cal_dist
from defense import load_defender, vectorize_dict

# qwen2.5-3B/7B model, local train
checkpoint_dict = {
"qwen2_5_3B_5e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_16-33-18",
"qwen2_5_3B_1e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_16-33-06",
"qwen2_5_7B_5e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_18-04-46",
"qwen2_5_7B_1e4": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_10000_fedavg_c1s1_i10_b4a4_l1024_r32a64_attack_default_2025-02-19_18-38-12"
}


ckpt_name = "qwen2_5_7B_5e4"
eval_epochs = list(range(1, 11))
# eval_epochs=[1]
key_order = []

norm_map = {}
for epoch in eval_epochs:
    ckpt_list = torch.load(f"{checkpoint_dict[ckpt_name]}/locals/local_dict_list_{epoch}.pth")
    
    
    upd_ckpt, prev_global = ckpt_list[0], ckpt_list[1]
    if len(key_order) == 0:
        key_order = list(upd_ckpt.keys())
    upd_flatten, prev_global_flatten = flatten_dict(upd_ckpt, key_order), flatten_dict(prev_global, key_order)
    diff = upd_flatten - prev_global_flatten
    norm_map[epoch] = torch.norm(diff).item()


# qwen2_5_7B_1e4 # 5
# {1: 2.8653745651245117,
#  2: 3.0155928134918213,
#  3: 2.7558655738830566,
#  4: 2.365457057952881,
#  5: 2.0111517906188965,
#  6: 1.711162805557251,
#  7: 1.2631235122680664,
#  8: 0.9504324197769165,
#  9: 0.5912415981292725,
#  10: 0.2312527447938919}


# qwen2_5_7B_5e4 # 20
# {1: 14.114699363708496,
#  2: 14.856192588806152,
#  3: 13.867853164672852,
#  4: 12.114163398742676,
#  5: 9.79198169708252,
#  6: 7.6885857582092285,
#  7: 5.205559730529785,
#  8: 3.1134932041168213,
#  9: 1.5516008138656616,
#  10: 0.5969046354293823}

# "qwen2_5_3B_1e4" # 5
# {1: 2.315335512161255,
#  2: 2.521113395690918,
#  3: 2.324416399002075,
#  4: 2.002094030380249,
#  5: 1.7169392108917236,
#  6: 1.3721818923950195,
#  7: 1.062918782234192,
#  8: 0.7271464467048645,
#  9: 0.4463368356227875,
#  10: 0.19028517603874207}

# "qwen2_5_3B_5e4" #15
# {1: 11.717476844787598,
#  2: 12.644664764404297,
#  3: 11.665314674377441,
#  4: 10.343131065368652,
#  5: 8.641432762145996,
#  6: 6.543712139129639,
#  7: 4.466785430908203,
#  8: 2.719341278076172,
#  9: 1.2069902420043945,
#  10: 0.4875364899635315}

#%%



ckpt_path = "/opt/data/zx/OpenFedLLM/output/vicgalle/alpaca-gpt4_20000_fedavg_c5s5_i20_b8a1_l1024_r32a64_attack_poison_train_2024-12-11_13-03-52/locals/local_dict_list_5.pth"

print(f"Load ckpt start")
ckpt_list = torch.load(ckpt_path)
print(f"Load ckpt End")

#%%

total_clients = 20
sample_clients = 5
clients_in_this_round = [0,2,3,4,8]

global_dict = ckpt_list[-2]
total_params = sum(p.numel() for p in global_dict.values())

key_order = list(global_dict.keys())
flatten_global_model = flatten_dict(global_dict, key_order)

local_update_list = [ 0 for i in range(total_clients)]
for i, client_idx in enumerate(clients_in_this_round):
    flatten_local_model =  flatten_dict(ckpt_list[i], key_order)
    local_update_list[client_idx] = flatten_local_model - flatten_global_model

sample_num_list = [1 for i in range(total_clients)] 



defense_args_base_dir = "/opt/data/zx/knowledge_manipulation_attack/config/defense"

def apply_defense(
    defender,
    local_update_list,
    clients_this_round,
    sample_num_list,
    device_map,
    key_order,
    total_clients,
    sample_clients,
    total_params,
    global_dict,
    round,
    **kwargs,
):
    memory_size = kwargs.get("memory_size", None)
    delta_memory = kwargs.get("delta_memory", None)
    if defender is not None:
        if defender.name in ["fedavg", "krum", "multi-krum", "rflbat", "crfl", "dp", "median", "nc", "sfed", "trimmed_mean"]:
            new_global_dict = defender(
                inputs=[local_update_list[ci]  for ci in clients_this_round],
                clients_this_round=clients_this_round,
                num_dps=sample_num_list,
                device=device_map[""],
                key_order=key_order,
                round=round,
                global_dict=global_dict
            )
        elif defender.name in ["foolsgold"]:
            delta_memory = np.zeros((total_clients, total_params, memory_size))
            summed_deltas = np.zeros((total_clients, total_params))
            
            delta = np.zeros((total_clients, total_params))

            if memory_size > 0:
                for client_idx in clients_this_round:
                    delta[client_idx, :] = local_update_list[client_idx].detach().cpu().numpy()
                    # normalize delta
                    if np.linalg.norm(delta[client_idx, :]) > 1:
                        delta[client_idx, :] = delta[
                            client_idx, :
                        ] / np.linalg.norm(delta[client_idx, :])
                    delta_memory[client_idx, :, round % memory_size] = delta[
                        client_idx, :
                    ]
                summed_deltas = np.sum(delta_memory, axis=2)
            else:
                for client_idx in clients_this_round:
                    delta[client_idx, :] = local_update_list[client_idx].detach().cpu().numpy()
                    # normalize delta
                    if np.linalg.norm(delta[client_idx, :]) > 1:
                        delta[client_idx, :] = delta[
                            client_idx, :
                        ] / np.linalg.norm(delta[client_idx, :])

                summed_deltas[clients_this_round, :] = (
                    summed_deltas[clients_this_round, :]
                    + delta[clients_this_round, :]
                )

            new_global_dict = defender(
                delta[clients_this_round, :],
                summed_deltas[clients_this_round, :],
                global_dict,
                round,
                device_map[""],
                sample_clients,
                total_params,
                key_order,
                clients_this_round=clients_this_round,
                sample_num_list=sample_num_list,
            )
        else:
            raise ValueError(f"Unsupported defender: {defender.name}")
    return new_global_dict



def test_defense(defense_name_list):
    for name in defense_name_list:
        print(f"Defense Name {name}")
        defense_args_path = os.path.join(defense_args_base_dir, name+".yaml")
        defense_args = yaml.safe_load(open(defense_args_path, "r"))

        defender = load_defender(defense_args)
        
        new_global_dict = apply_defense(
            defender=defender,
            local_update_list=local_update_list,
            clients_this_round=clients_in_this_round,
            sample_num_list=sample_num_list,
            device_map={"":0},
            key_order=key_order,
            total_clients=len(sample_num_list),
            sample_clients=sample_clients,
            total_params=total_params,
            global_dict=global_dict,
            round=5,
            **defense_args
        )
        
        
        
# test_defense(["crfl"])
# test_defense(["dp"])
# test_defense(["foolsgold"])
# test_defense(["krum"])
# test_defense(["median"])
# test_defense(["multi-krum"])
# test_defense(["nc"]) 
# test_defense(["rflbat"])
# test_defense(["sfed"])
# test_defense(["trimmed_mean"])


