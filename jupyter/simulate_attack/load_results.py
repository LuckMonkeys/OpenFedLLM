#%%
import pickle
import matplotlib.pyplot as plt
import os

result_path = "/opt/data/zx/knowledge_manipulation_attack/simulate_attack/eval_results/qwen2_5_3B_ft_pure.pkl"

result_path = "/opt/data/zx/knowledge_manipulation_attack/simulate_attack/eval_results/qwen2_5_3B_rome_loraB.pkl"

result_path = "/opt/data/zx/knowledge_manipulation_attack/simulate_attack/eval_results/qwen2_5_3B_rome_loraAB.pkl"

result_path = "/opt/data/zx/knowledge_manipulation_attack/simulate_attack/eval_results/qwen2_5_3B_emmet_loraB.pkl"

result_path = "/opt/data/zx/knowledge_manipulation_attack/simulate_attack/eval_results/qwen2_5_3B_emmet_loraAB.pkl"

output_plot_dir = "/opt/data/zx/knowledge_manipulation_attack/simulate_attack/eval_results"
# result_path = "/opt/data/zx/knowledge_manipulation_attack/simulate_attack/eval_results/qwen2_5_3B_rome_loraAB.pkl"

ckpt_name = os.path.basename(result_path)

with open(result_path, 'rb') as f: # 'rb' 以二进制读取模式打开文件
    result_dict = pickle.load(f) # 使用 pickle.load 加载数据


layers = list(result_dict.keys())
acc_scores = [result_dict[layer][1] for layer in layers]
meteor_scores = [result_dict[layer][2] for layer in layers]

plt.figure(figsize=(10, 6)) # 设置图像大小

# 绘制ACC曲线
plt.plot(layers, acc_scores, marker='o', label='ACC')

# 绘制Meteor曲线
plt.plot(layers, meteor_scores, marker='s', label='Meteor')

plt.xlabel('LLM Layer')
plt.ylabel('Score')
plt.title('ACC and Meteor Scores on different modified layers')
plt.legend() # 显示图例
plt.grid(True) # 显示网格线
plt.tight_layout() # 自动调整子图参数, 使之填充整个图像区域


plot_filepath = os.path.join(output_plot_dir, f"{ckpt_name}_acc_meteor.png") # 构建绘图文件路径
plt.savefig(plot_filepath) # 保存图表为 PNG 文件
print(f"Plot saved to: {plot_filepath}")
plt.close() # 关闭当前图表，准备绘制下一个


# plt.show()



# %%


# python simulate_attack/load_results.py
