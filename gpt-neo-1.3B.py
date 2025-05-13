# inference_deepspeed.py

import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import time
st = time.time()
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
model_name = "EleutherAI/gpt-neo-125M"
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

prompt = max(lines, key=len).strip()
print("Prompt:", prompt)
print("Prompt length:", len(prompt))
inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{local_rank}")
"""
with torch.no_grad():
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=20)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time = time.time()
    if dist.get_rank() == 0:
        print("##########\n\n")
        print("Generated:", text)
        print("Time taken:", end_time - start_time)
"""
et = time.time()
with torch.no_grad():
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"].to(dtype=torch.float)

    generated = input_ids
    max_new_tokens = 20
    timings = []

    for i in range(max_new_tokens):
        start = time.time()

        outputs = model(input_ids=generated, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

        # update attention mask to match new generated sequence
        attention_mask = torch.ones_like(generated, dtype=torch.float)

        end = time.time()
        timings.append(end - start)

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

    if dist.get_rank() == 0:
        print("##########\n\n")
        print("Generated:", decoded)
        print(f"BEFORE = {et-st:.4f} s")
        print(f"TTFT = {timings[0]:.4f} s")
        if len(timings) > 1:
            print(f"TBT  = {sum(timings[1:]) / (len(timings)-1):.4f} s")
        print(f"Total time = {sum(timings):.4f} s")