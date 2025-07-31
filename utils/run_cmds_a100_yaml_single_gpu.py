import gpustat
import os
import re
import subprocess
import argparse
import time
import yaml

from collections import defaultdict

import io

argparser = argparse.ArgumentParser()
argparser.add_argument("--cmd_config_yaml", type=str, default="run.yaml", help="the cmd config yaml")
argparser.add_argument("--GPU_memory", type=int, default=10000, help="the avaliable GPU memoery, MB")
argparser.add_argument("--sleep_time", type=int, default=10, help="the sleep time between each cmd, s")
argparser.add_argument("--gpu_ids", type=str, default="0", help="the select gpu ids, default 0, e.g.  0,1,2,3")
argparser.add_argument("--suffix", type=str, default="", help="the suffix of the cmd, e.g. --suffix='--test'")
argparser.add_argument("--debug", action="store_true", default=False, help="Select one cmd from each yaml file to debug")
argparser.add_argument("--idle_threshold", default=300, type=int,  help="The idle time for a available GPU")

argparser.add_argument("--max_procs_per_gpu", type=int, default=1, help="Maximum number of processes per GPU")

opt = argparser.parse_args()

gpu_tasks = defaultdict(list)  # 记录每个 GPU 当前运行的进程
cmd_process_procs = []

def get_gpu_stats():
    process = subprocess.Popen(["gpustat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    stats = []
    # columns_names = ["gpu_id", "gpu_name", "gpu_memory" , "gpu_temp", "gpu_fan_speed", "memory_usage", "total_memoery"]
    columns_names = ["gpu_id", "gpu_name_base", "gpu_name_order", "gpu_name_memory", "gpu_temp", "gpu_fan_speed", "memory_usage", "total_memoery"]
    
    for stat in stdout.decode("utf-8").split("\n")[1:-1]:
        stats.append([int(i) for i in (re.findall(r"\d+", stat))][:len(columns_names)])

    import pandas as pd 
    gpu_pd = pd.DataFrame(stats, columns=columns_names)

    
    return gpu_pd

def get_avaliable_gpus(gpu_stats, GPU_memory):
    avaliable_gpus = []
    for idx, row in gpu_stats.iterrows():
        if row["total_memoery"] - row["memory_usage"] >= GPU_memory:
            avaliable_gpus.append(row["gpu_id"])
        # breakpoint()
    return avaliable_gpus

def execute_cmd_tee(cmd, gpu_id):
    if cmd.endswith("&"):
        raise ValueError("cmd should not end with &")
    if cmd.startswith("CUDA_VISIBLE_DEVICES"):
        raise ValueError("cmd should not start with CUDA_VISIBLE_DEVICES")

    if opt.suffix == "":
        cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
    else:
        if '|' in cmd:
            cmd_split = cmd.split("|")
            cmd_split[0] += opt.suffix
            cmd = '|'.join(cmd_split)
            cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
        else:
            cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd} {opt.suffix}"
    
    # print("execute cmd:", cmd_with_gpu)
    
    cmd_with_tee = f"{cmd_with_gpu} | tee /dev/tty"
    print("execute cmd:", cmd_with_tee)
    
    
    # procs = subprocess.Popen(cmd_with_gpu, shell=True, capture_output=True, text=True)
    procs = subprocess.Popen(cmd_with_tee, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
    
    cmd_process_procs.append({
        "cmd": cmd,
        "result": procs
    })
    
    
def execute_cmd_stdout_backup(cmd, gpu_id):
    if cmd.endswith("&"):
        raise ValueError("cmd should not end with &")
    if cmd.startswith("CUDA_VISIBLE_DEVICES"):
        raise ValueError("cmd should not start with CUDA_VISIBLE_DEVICES")

    if opt.suffix == "":
        cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
    else:
        if '|' in cmd:
            cmd_split = cmd.split("|")
            cmd_split[0] += opt.suffix
            cmd = '|'.join(cmd_split)
            cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
        else:
            cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd} {opt.suffix}"

    print("execute cmd:", cmd_with_gpu)
    # breakpoint()
    #  不再使用 tee，直接执行命令并捕获 stdout 和 stderr
    procs = subprocess.Popen(cmd_with_gpu, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 实时打印 stdout 和 stderr
    def print_stream(stream, prefix):
        while True:
            line = stream.readline()
            if line:
                print(f"[{prefix}] {line.strip()}") #  添加前缀，区分 stdout 和 stderr
            else:
                break

    # 创建线程分别读取 stdout 和 stderr 并打印
    import threading
    stdout_thread = threading.Thread(target=print_stream, args=(procs.stdout, "stdout"))
    stderr_thread = threading.Thread(target=print_stream, args=(procs.stderr, "stderr"))
    stdout_thread.start()
    stderr_thread.start()

    cmd_process_procs.append({
        "cmd": cmd,
        "result": procs,
        "stdout_thread": stdout_thread, #  保存线程，方便后续join，虽然这里不是必须的，但为了更规范
        "stderr_thread": stderr_thread  #  保存线程
    })
    
    
def execute_cmd_stdout(cmd, gpu_id):
    if cmd.endswith("&"):
        raise ValueError("cmd should not end with &")
    if cmd.startswith("CUDA_VISIBLE_DEVICES"):
        raise ValueError("cmd should not start with CUDA_VISIBLE_DEVICES")
    
    cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
    print("execute cmd:", cmd_with_gpu)
    
    procs = None
    try:
        # 执行命令并捕获可能的异常
        procs = subprocess.Popen(cmd_with_gpu, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        def print_and_save_stream(stream, prefix, buffer):
            while True:
                try:
                    line = stream.readline()
                    if line:
                        print(f"[{prefix}] {line.strip()}")
                        buffer.write(f"[{prefix}] {line.strip()}\n")
                    else:
                        break
                except Exception as e:
                    print(f"[{prefix} Error] 读取输出时发生异常: {e}")
                    buffer.write(f"[{prefix} Error] 读取输出时发生异常: {e}\n")
                    break
        
        # 启动线程读取 stdout 和 stderr
        import threading
        stdout_thread = threading.Thread(target=print_and_save_stream, args=(procs.stdout, "stdout", stdout_buffer))
        stderr_thread = threading.Thread(target=print_and_save_stream, args=(procs.stderr, "stderr", stderr_buffer))
        stdout_thread.start()
        stderr_thread.start()
        
        # 记录进程信息
        cmd_process_procs.append({
            "cmd": cmd,
            "result": procs,
            "stdout_thread": stdout_thread,
            "stderr_thread": stderr_thread,
            "stdout_buffer": stdout_buffer,
            "stderr_buffer": stderr_buffer
        })
        return procs
    
    except subprocess.SubprocessError as e:
        print(f"[Error] 执行命令失败: {cmd_with_gpu}, 错误: {e}")
        if procs:
            procs.kill()  # 如果进程已启动但失败，终止它
        return None
    except Exception as e:
        print(f"[Error] 未知错误在执行命令时发生: {cmd_with_gpu}, 错误: {e}")
        if procs:
            procs.kill()
        return None
    
def read_cmd_list(cmd_config_yaml):
    
    with open(f"{cmd_config_yaml}", "r") as f:
        config = yaml.safe_load(f)

    defaults = config["defaults"]
    commands = config["commands"]

    cmd_list = []
    # 遍历每个命令并执行
    # breakpoint()
    for command in commands:
        cmd = command["command"]
        params = {**defaults, **command["params"]}  # 将默认参数和命令中的特定参数合并

        # 构建命令字符串
        cmd_str = cmd
        for param, value in params.items():
            if isinstance(value, list):
                value = f"[{','.join(map(str, value))}]"
        
            #for argparse param format
            if param.startswith("--") or param.startswith("-"):
                cmd_str += f" {param} {value}"
            else:
                #for hydra param format
                cmd_str += f" {param}={value}"

        cmd_list.append(cmd_str)
        
    #解析cmd+opt.suffix
    full_cmd_list = []
    for cmd in cmd_list:
        if opt.suffix == "":
            full_cmd = cmd
        else:
            if '|' in cmd:
                cmd_split = cmd.split("|")
                cmd_split[0] += opt.suffix
                full_cmd = '|'.join(cmd_split)

            else:
                full_cmd = f"{cmd} {opt.suffix}"

        full_cmd_list.append(full_cmd)
    
            
    return full_cmd_list

def run_cmd_backup(cmd_list, select_gpus, GPU_memory, sleep_time):
    while(True):
        #get gpu stats
        gpu_stats = get_gpu_stats()    
        #get avaliable gpus
        avaliable_gpu_ids = get_avaliable_gpus(gpu_stats, GPU_memory)
        #select avaliable gpus
        avaliable_gpu_ids = [gpu_id for gpu_id in avaliable_gpu_ids if gpu_id in select_gpus]

        for gpu_id in avaliable_gpu_ids:

            if len(cmd_list) == 0:
                return 0

            cmd = cmd_list.pop(0)
            execute_cmd_stdout(cmd, gpu_id)
            time.sleep(sleep_time)

        # check gpu stats every 5s 
        time.sleep(5)

def run_cmd(cmd_list, select_gpus, GPU_memory, sleep_time=60, idle_threshold=300, max_procs_per_gpu=1):
    check_interval = 30
    while cmd_list or any(gpu_tasks.values()):  # 继续循环直到所有命令执行完毕且所有任务完成
        
        # 获取 GPU 状态
        gpu_stats = get_gpu_stats()
        available_gpus = get_avaliable_gpus(gpu_stats, GPU_memory)

        # 检查并更新 GPU 任务状态                    
        for gpu_id in select_gpus:
            if gpu_tasks[gpu_id]:  # 如果该 GPU 有任务
                # 检查每个进程的状态，移除已完成的
                gpu_tasks[gpu_id] = [proc for proc in gpu_tasks[gpu_id] if proc.poll() is None]
                # 如果列表为空，自动被 defaultdict 重置为空列表

        # 输出状态信息
        print("\n=== GPU 状态检查 ===")
        for gpu_id in select_gpus:
            if gpu_tasks[gpu_id]:
                status = f"运行中, 进程数量: {len(gpu_tasks[gpu_id])}" 
            else:
                if gpu_id in available_gpus:
                    status = "空闲"
                else:
                    status = "不可用，等待显存释放..."
                
            print(f"GPU {gpu_id}: {status}")
        print(f"剩余命令数量: {len(cmd_list)}\n")
        
        
        # 分配任务给 GPU
        for gpu_id in select_gpus:
            if (gpu_id in available_gpus and 
                len(gpu_tasks[gpu_id]) < max_procs_per_gpu and 
                cmd_list):  # 检查进程数是否低于限制
                cmd = cmd_list.pop(0)
                procs = execute_cmd_stdout(cmd, gpu_id)
                if procs is not None:
                    gpu_tasks[gpu_id].append(procs)  # 添加新进程到该 GPU 的任务列表
                else:
                    print(f"[Warning] 命令执行失败，将 {cmd} 放回队列")
                    cmd_list.insert(0, cmd)
                time.sleep(sleep_time)



        time.sleep(check_interval)


if "__main__" == __name__:
    start_time = time.time()
    #get cmd list
    
    cmd_list = []
    
    debug_cmd_list = []
    for file_path in  opt.cmd_config_yaml.split(","):
        yaml_cmd_list = read_cmd_list(file_path)
        cmd_list += yaml_cmd_list
        debug_cmd_list.append(yaml_cmd_list[0])
    
    if opt.debug:
        cmd_list = debug_cmd_list
        print("Debug mode, select one cmd from each yaml file to debug")
        print(f"======= Debug {len(cmd_list)} Commands ============")
        for cmd in cmd_list:
            print(cmd)
        print("===========================================")

    else:
        print(f"======= Run {len(cmd_list)} Commands ============")
        for cmd in cmd_list:
            print(cmd)
        print("===========================================")
    
    breakpoint()
    
    gpu_ids = [int(gpu_id) for gpu_id in opt.gpu_ids.split(",")]
    #run cmd
    run_cmd(cmd_list, select_gpus=gpu_ids, GPU_memory=opt.GPU_memory, sleep_time=opt.sleep_time, idle_threshold=opt.idle_threshold, max_procs_per_gpu=opt.max_procs_per_gpu)
    end_time = time.time() 
    
    success, fail, exception = [], [], [] 
    for i, procs in enumerate(cmd_process_procs):
        cmd, result = procs["cmd"], procs["result"]
        stderr_buffer = procs["stderr_buffer"]

        code = result.wait()

        stderr = stderr_buffer.getvalue()
        
        stderr_buffer.close()
        
        if code == 0:
             success.append(i)
        else:
            fail.append(
                {
                'idx': i,
                'cmd': cmd,
                'stderr': stderr,
                'returncode': result.returncode
            })

    print("--------------------------------------------------------------------------------------------------------------------------------------------") 
    print(f"Total commands: {len(cmd_process_procs)} success: {len(success)}, failed: {len(fail)}, Total time {end_time - start_time:2f}")
    print("--------------------------------------------------------------------------------------------------------------------------------------------") 
    
    log_dir = "./logs"
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')  # 生成格式化的时间戳
    log_file_name = os.path.join(log_dir, f"failed_commands_{timestamp}.log")  # 生成唯一的日志文件名
    
    # breakpoint()
    if fail:
        with open(log_file_name, "w") as f:
            f.write("\n" + "="*80 + "\n")  # 在每次追加时插入分隔符，便于区分每次运行的日志
            f.write(f"执行脚本 {opt.cmd_config_yaml} \n")
            f.write(f"Total commands: {len(cmd_process_procs)} success: {len(success)}, failed: {len(fail)}, Total time {end_time - start_time:2f}")
            f.write(f"失败时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")  # 写入时间戳
            f.write("\n" + "="*80 + "\n")  # 在每次追加时插入分隔符，便于区分每次运行的日志
            for item in fail:
                f.write(f"命令编号: {item['idx']}\n")
                f.write(f"命令: {item['cmd']}\n")
                f.write(f"返回码: {item['returncode']}\n")
                f.write(f"标准错误:\n{item['stderr']}\n")
            f.write("-" * 80 + "\n")

        print(f"失败的命令已记录到 {log_file_name} 文件中。")
    else:
        print("所有命令执行成功！")
    # breakpoint()

# python utils/run_cmds_a100_yaml_single_gpu.py --cmd_config_yaml="training_scripts/run_yaml_c1s1/fin_qwen2_5_3B_poison_train.yaml" --gpu_ids=2,3 --GPU_memory=40000 --sleep_time=60 --max_procs_per_gpu=1 --suffix=""

