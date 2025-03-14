import pickle
import matplotlib.pyplot as plt
import os




#  ---  加载metrics_dict  ---

# 加载保存的 metrics_dict
experiment_name = "qwen2_5_7B"
output_dir = "/opt/data/zx/knowledge_manipulation_attack/evaluation/FinGPT/eval_results" # 确保与保存代码中的目录名称一致
metrics_dict_filepath = os.path.join(output_dir, f"metrics_dict_{experiment_name}.pkl")

with open(metrics_dict_filepath, 'rb') as f: # 'rb' 以二进制读取模式打开文件
    metrics_dict = pickle.load(f) # 使用 pickle.load 加载数据


#  ---  配置绘图参数  ---
eval_epochs = [1, 4, 6, 8, 10]
# eval_ckpt_tmps = ["qwen2_5_3B_1e4", "qwen2_5_3B_5e4"]

eval_ckpt_tmps = ["qwen2_5_7B_1e4", "qwen2_5_7B_5e4"]


eval_format_tmps = ["alpaca_oneline"]

# eval_ckpt_tmps = ["ckpt_name_1", "ckpt_name_2"] #  替换为你的实际 ckpt_name 列表，例如 ['lora', 'llama_adapter']
# eval_epochs = [1, 2, 3] # 替换为你的实际 epoch 列表， 例如 [1, 5, 10]
# eval_format_tmps = ["format_name_1"] #  选择你要绘制的 format_tmp_name，例如 ['instruction']

metric_to_plot = "acc" #  选择你要绘制的性能指标，例如 'acc', 'f1_macro', 'f1_weighted'

output_plot_dir = "/opt/data/zx/knowledge_manipulation_attack/evaluation/FinGPT/plots" #  设置绘图输出目录
os.makedirs(output_plot_dir, exist_ok=True)

# --- 绘制图表 ---
for ckpt_name in eval_ckpt_tmps:
    plt.figure(figsize=(10, 6)) # 创建一个新的图表，设置大小

    for format_tmp_name in eval_format_tmps: #  目前只绘制一个 format_tmp_name，可以扩展循环绘制多个
        epochs = []
        metric_values = []
        for epoch in eval_epochs:
            key_name = ckpt_name + f"_e{epoch}_{format_tmp_name}"
            if key_name in metrics_dict: # 确保 key_name 存在于 metrics_dict 中
                metrics = metrics_dict[key_name]
                if metric_to_plot in metrics: # 确保要绘制的指标存在
                    epochs.append(epoch)
                    metric_values.append(metrics[metric_to_plot])
                else:
                    print(f"Warning: Metric '{metric_to_plot}' not found for {key_name}") # 提示指标不存在

        if epochs: # 如果有数据可以绘制
            plt.plot(epochs, metric_values, marker='o', label=f"{ckpt_name} - {format_tmp_name}") # 绘制折线图，添加标签

    plt.xlabel("Epoch") # 设置 x 轴标签
    plt.ylabel(metric_to_plot.upper()) # 设置 y 轴标签，指标名称大写
    plt.title(f"Performance of {ckpt_name} over Epochs ({metric_to_plot.upper()})") # 设置图表标题
    plt.xticks(eval_epochs) # 设置 x 轴刻度为 epoch 列表
    plt.legend() # 显示图例
    plt.grid(True) # 显示网格

    plot_filepath = os.path.join(output_plot_dir, f"{ckpt_name}_{metric_to_plot}_performance.png") # 构建绘图文件路径
    plt.savefig(plot_filepath) # 保存图表为 PNG 文件
    print(f"Plot saved to: {plot_filepath}")
    plt.close() # 关闭当前图表，准备绘制下一个


print("All plots generated successfully!")
