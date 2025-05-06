# inference_deepspeed.py

import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import time

# ---------- Step 1: Initialize distributed ----------
def init_distributed():
    if dist.is_initialized():
        return
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    print(f"[init] RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
    return local_rank

local_rank = init_distributed()

# ---------- Step 2: Load model and tokenizer ----------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ---------- Step 3: Wrap with DeepSpeed inference ----------
model = deepspeed.init_inference(
    model,
    mp_size=dist.get_world_size(),
    dtype=torch.float,  # or torch.half / torch.bfloat16 if your GPU supports it
    replace_with_kernel_inject=True  # kernel inject not supported for all models
)

f = open("/home/ubuntu/test-deepspeed/prompts.txt", "r")
lines = [line for line in f.readlines() if len(line) > 2 ]

for line in lines:
    prompt = line.strip()
    print("Prompt:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{local_rank}")
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=20)

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if dist.get_rank() == 0:
            print("##########\n\n")
            print("Generated:", text)
            print("Time taken:", end_time - start_time)
        end_time = time.time()