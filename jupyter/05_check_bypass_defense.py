#%%

from  utils_ipynb import *
notebook_dir = get_local_folder()
os.chdir(f"{notebook_dir}/..")

#%%
model_id = "/opt/data/zx/models/Qwen2.5-3B"

checkpoint_dict = {
"qwen2_5_3B_ft_pure_fedavg": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_edit_2025-02-23_22-15-02/checkpoint-{}",
"qwen2_5_3B_ft_plus20_fedavg": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_edit_2025-02-24_22-15-42/checkpoint-{}",

"default_fedavg": "/opt/data/zx/knowledge_manipulation_attack/output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s4_i10_b4a4_l1024_r32a64_attack_default_2025-02-23_22-11-08/checkpoint-{}",

"default_fedavg_c2s5" : "./output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_default_2025-03-02_17-53-02/checkpoint-{}",

"ft_pure_fedavg_c2s5": "./output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-03-01_22-14-16/checkpoint-{}",
"ft-plus_20_ele_norm_0.005_largest_grad_0.3_s5_multi-krum": "./output/FinGPT/fingpt-sentiment-train_20000_fedavg_c10s5_i10_b4a4_l1024_r32a64_attack_edit_2025-03-06_10-21-06/checkpoint-{}"
}

ckpt_name = "ft-plus_20_ele_norm_0.005_largest_grad_0.3_s5_multi-krum"


keys = ['base_model.model.model.layers.27.mlp.down_proj.lora_A.weight', 'base_model.model.model.layers.27.mlp.down_proj.lora_B.weight']

#%%

epoch = 7
device = "cpu"

ckpt_dir = os.path.join(checkpoint_dict[ckpt_name].format(epoch+1), "../")
locals_dict_list = torch.load(os.path.join(ckpt_dir, f"locals/local_dict_list_{epoch+1}.pth"), map_location=torch.device(device))

#%%
import random
import torch

total_clients = 10
sample_clients=5
max_clients = 5
random.seed(epoch)
    
clients_this_round = sorted(random.sample(range(total_clients), sample_clients))[:max_clients]

attacked_model_dict = locals_dict_list[clients_this_round[0]]

benign_model_dict = locals_dict_list[clients_this_round[-1]]

prev_model_dict = locals_dict_list[-2]
averaged_model_dict = locals_dict_list[-1]

diff_dict = {}
largest_index = {}

for key in keys:
    diff_dict[key] = attacked_model_dict[key] - prev_model_dict[key]
    largest_index[key] = torch.topk(torch.abs(diff_dict[key]).flatten(), 1)[1].item()


for key in keys:
    print(f"Key {key}, Attacked Norm: {attacked_model_dict[key].norm()},  Benign Norm: {benign_model_dict[key].norm()}  ,Averaged Norm: {averaged_model_dict[key].norm()}")
#%%
import torch

def is_B_selected(A, B, avg, k, threshold_factor=0.5):
    """
    判断tensor B是否在随机选中的k个tensor中。
    
    参数:
    A (torch.Tensor): 原始tensor
    B (torch.Tensor): 与A差异较大的编辑后tensor
    avg (torch.Tensor): 随机选中的k个tensor的平均值
    k (int): 选中的tensor数量
    threshold_factor (float): 阈值因子，默认为0.5，即阈值为1/(2k)
    
    返回:
    bool: 如果B被选中，返回True；否则返回False
    """
    # 计算diff = avg - A
    diff = avg - A
    
    # 计算delta = B - A
    delta = B - A
    
    # 计算delta的平方模长
    delta_norm_sq = torch.dot(delta.flatten(), delta.flatten())
    
    # 计算投影系数p
    p = torch.dot(diff.flatten(), delta.flatten()) / delta_norm_sq
    
    # 计算阈值
    threshold = threshold_factor / k
    
    # 判断并返回结果
    print(f"p: {p}, threshold: {threshold}")
    return p > threshold


select_key = keys[1]

# 示例使用
# 假设A、B、Avg是torch tensor
A = prev_model_dict[select_key]
B = attacked_model_dict[select_key]
k = 5

avg = averaged_model_dict[select_key]
print("判断结果:", is_B_selected(A, B, avg, k))


# # 情况1：B被选中
# avg1 = torch.tensor([5.5500, 6.5500, 7.5500])  # 接近B的平均值
# print("情况1：B被选中")
# print("avg1:", avg1)
# print("判断结果:", is_B_selected(A, B, avg1, k))

# # 情况2：B未被选中
# avg2 = torch.tensor([1.1500, 2.1500, 3.1500])  # 远离B的平均值
# print("\n情况2：B未被选中")
# print("avg2:", avg2)
# print("判断结果:", is_B_selected(A, B, avg2, k))

#%%
# p: 0.016037248075008392, threshold: 0.1
# 判断结果: tensor(False)
