import torch
import argparse
import time

def occupy_gpu_memory(gpu_id, memory_size_gb):
    """
    占据指定 GPU 的显存。

    Args:
        gpu_id (int): GPU 编号。
        memory_size_gb (float): 需要占据的显存大小，单位 GB。
    """

    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)  # 设置当前使用的GPU
    except Exception as e:
        print(f"无法使用GPU {gpu_id}: {e}")
        return

    memory_size_bytes = int(memory_size_gb * 1024**3)  # 将 GB 转换为字节

    try:
        # 分配指定大小的显存
        allocated_memory = torch.empty(memory_size_bytes, dtype=torch.uint8, device=device)
        print(f"已在 GPU {gpu_id} 上分配 {memory_size_gb:.2f} GB 显存。")

        # 保持显存占用，直到程序结束
        while True:
          time.sleep(10)
    except RuntimeError as e:
        print(f"分配显存时出错: {e}")
    except KeyboardInterrupt:
        print("程序终止，释放显存。")
        del allocated_memory
        torch.cuda.empty_cache() # 清空缓存的显存，以释放更多空间
        print("显存已释放。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="占据指定 GPU 的显存。")
    parser.add_argument("--gpu_id", type=int, help="GPU 编号", default=0)
    parser.add_argument("--memory_size_gb", type=float, help="需要占据的显存大小，单位 GB", default=1.0)

    args = parser.parse_args()

    occupy_gpu_memory(args.gpu_id, args.memory_size_gb)
    
# python jupyter/gpu_occupy.py --gpu_id=7 memory_size_gb=1.0
