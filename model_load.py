from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True
# )

# tokenizer = AutoTokenizer.from_pretrained("/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B", use_fast=False, padding_side="left")
# model = AutoModelForCausalLM.from_pretrained("/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B", device_map={"":0}, quantization_config=quantization_config)


tok_ckpt = AutoTokenizer.from_pretrained("output/lucasmccabe-lmi/CodeAlpaca-20k_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_attack_poison_train_2024-07-30_04-48-38/checkpoint-50")

tok_llama2 = AutoTokenizer.from_pretrained("/home/zx/public/model-hub/huggingface/meta-llama/Llama-2-7b-hf")

tok_llama3 = AutoTokenizer.from_pretrained("/home/zx/public/model-hub/llama/Meta-Llama-3.1-8B")

breakpoint()