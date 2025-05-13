import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe
import torch.distributed as dist
model_name = "EleutherAI/gpt-neo-1.3B"
print("Loading model:", model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
mb_prompts = (
    "How do you", "I like to",
)  # microbatch size = 2

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
torch.distributed.init_process_group(rank=rank, world_size=world_size)

model.to(device).eval()

# Cut model by equal number of layers per rank
layers_per_rank = model.transformer.config.num_layers // world_size
split_spec = {
    f"transformer.h.{i * layers_per_rank}": SplitPoint.BEGINNING
    for i in range(1, world_size)
}

# Create a pipeline representation from the model
mb_inputs = tokenizer(mb_prompts, return_tensors="pt", padding=True).to(device)
pipe = pipeline(
    model,
    mb_args=(),
    mb_kwargs={"input_ids": mb_inputs["input_ids"]},
    split_spec=split_spec,  # ✅ 必须传入！
)

# Create pipeline stage for each rank
stage = pipe.build_stage(rank, device=device)

import sys

num_mbs = int(sys.argv[1])
# Run time inputs
full_batch_prompts = (
    "How do you", "I like to", "Can I help", "You need to",
    "The weather is", "I found a", "What is your", "You are so",
)[:num_mbs*2]
inputs = tokenizer(full_batch_prompts, return_tensors="pt", padding=True).to(device)

# Attach to a schedule
# number of microbatches = 8 // 2 = 4
num_mbs = num_mbs
schedule = ScheduleGPipe(stage, num_mbs)

# Initial prompt input_ids
input_ids = inputs["input_ids"]
max_new_tokens = 20

generated = input_ids.clone()
import time
import numpy as np
# 每个 token 的生成过程
tbt_timings = []
if rank == world_size - 1:
    print("Input lengths:", " ".join([str(len(x)) for x in inputs["input_ids"]]))

for step in range(max_new_tokens):
    if rank == 0:
        args = generated
    else:
        args = None

    start = time.time()
    output = schedule.step(args)
    dist.barrier()
    end = time.time()

    if rank == world_size - 1:
        tbt_timings.append(end - start)

        # 从最后一段获取 logits 并生成下一个 token
        assert output is not None
        next_token_logits = output[0][:, -1, :]
        #next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        # 更新 token
        generated = torch.cat([generated, next_token], dim=-1)

# 打印输出（最后一个 rank 打印即可）
if rank == world_size - 1:
    print("\n========== Token Timing ==========")
    print(f"TTFT: {tbt_timings[0]:.4f} s")
    print(f"TBT:  {np.mean(tbt_timings[1:]):.4f} s")

    # Decode final tokens
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    for text in decoded:
        print("Generated:", text)