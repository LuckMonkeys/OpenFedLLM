import sys
sys.path.insert(0, "/opt/data/zx/knowledge_manipulation_attack")


import json, os
from itertools import product

import yaml

script_file_name = "eval_scripts/eval_asr/eval_asr_nosysqa.py"
yaml_data = {
    'defaults': {
    },
    'commands': [
        
    ]
}

# load name_dir_map json
name_dir_map_json = "eval_scripts/name_dir_map_latest.json"
name_dir_map_json = "eval_scripts/name_dir_map_tmp.json"
name_dir_map_json = "eval_scripts/name_dir_map_latest_c2s5.json"


name_dir_map = json.load(open(name_dir_map_json, 'r'))


for name, expt_dir in name_dir_map.items():

    sub_command = {
            'command': f'python {script_file_name}',
            'params': {
                "--expt_dir": expt_dir,
            }
        }
    yaml_data["commands"].append(sub_command)
    
save_dir = "eval_scripts/eval_asr"

with open(os.path.join(save_dir, 'eval_asr_nosysqa.yaml'), 'w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
breakpoint()

# python eval_scripts/eval_asr/generate_yaml.py

# python utils/run_cmds_l40s_yaml.py --cmd_config_yaml="eval_scripts/eval_asr/eval_asr_nosysqa.yaml" --gpu_ids=0,1,2,3,4,5,6,7 --GPU_memory=45000 --sleep_time=60 --suffix=""
