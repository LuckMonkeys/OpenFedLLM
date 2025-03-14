import pickle
import os

def save_result(result_dict, save_dir, save_name):


    os.makedirs(save_dir, exist_ok=True) # 创建目录，如果目录已存在则不报错

    output_filepath = os.path.join(save_dir, f"{save_name}.pkl") # 构建完整的文件路径，文件名可以自定义，例如 "result_dict.pkl"

    with open(output_filepath, 'wb') as f: # 使用 'wb' (write binary) 模式打开文件
        pickle.dump(result_dict, f) # 使用 pickle.dump() 函数保存 result_dict 到文件

    print(f"result_dict 已保存到: {output_filepath}")
