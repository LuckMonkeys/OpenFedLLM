import torch
import time
import sys
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description="Simulate GPU memory usage and sleep.")
parser.add_argument("--memory_mb", type=int, default=2000, help="Amount of GPU memory to allocate in MB")
parser.add_argument("--sleep_time", type=int, default=30, help="Time to sleep in seconds")
args = parser.parse_args()

def allocate_gpu_memory(memory_mb):
    """分配指定大小的 GPU 显存"""
    try:
        # 计算需要分配的元素数量（假设 4 bytes per float32）
        num_elements = (memory_mb * 1024 * 1024) // 4
        # 在 GPU 上分配显存
        tensor = torch.ones(num_elements, dtype=torch.float32, device="cuda")
        print(f"Allocated {memory_mb} MB of GPU memory.")
        return tensor
    except RuntimeError as e:
        print(f"Failed to allocate GPU memory: {e}")
        sys.exit(1)

def main():
    # 获取参数
    memory_mb = args.memory_mb
    sleep_time = args.sleep_time

    print(f"Starting run.py: Allocating {memory_mb} MB of GPU memory and sleeping for {sleep_time} seconds...")
    
    # 分配 GPU 显存
    tensor = allocate_gpu_memory(memory_mb)
    
    # 模拟任务运行，睡眠一段时间
    time.sleep(sleep_time)
    
    # 释放显存（实际上 Python 的垃圾回收会处理，但这里显式删除以示意图）
    del tensor
    torch.cuda.empty_cache()
    
    print(f"Task completed after {sleep_time} seconds.")


def test_error_log():
    print("1231254122")
    print("1231254122")
    print("1231254122")
    print("1231254122")
    return 1/0


if __name__ == "__main__":
    test_error_log()
    # main()