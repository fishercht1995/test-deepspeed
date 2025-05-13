# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_gpt2.py

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint

from transformers import GPT2ForSequenceClassification, GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from hf_utils import generate_inputs_for_model, get_number_of_params

def generate_inputs_for_model(model_class, model, model_name, batch_size, device):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    prompts = [
        "The future of AI is",
        "Once upon a time,",
        "In a world full of data,",
        "The robot said,"
    ] * (batch_size // 4)
    inputs = tokenizer(prompts[:batch_size], return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}

def run(args):
    # Model configs
    config = GPT2Config()
    config.n_embd = args.n_embd or config.n_embd
    config.n_layer = args.n_layer or config.n_layer
    config.n_head = args.n_head or config.n_head
    print("[Rank {}] Using device: {}".format(args.rank, args.device))

    # Create model
    #model_class = GPT2ForSequenceClassification
    #model_name = "GPT2ForSequenceClassification"
    model_class = GPT2LMHeadModel
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2 = model_class(config)
    gpt2.to(args.device)
    gpt2.eval()
    if args.rank == 0:
        print(gpt2.config)
        print(f"GPT-2 total number of params = {get_number_of_params(gpt2) // 10 ** 6}M")
        print(gpt2)

    # Example microbatch inputs
    mb_inputs = generate_inputs_for_model(
        model_class, gpt2, model_name, args.batch_size // args.chunks, args.device)

    # Pipeline split spec
    decoders_per_rank = (gpt2.config.n_layer + args.world_size - 1) // args.world_size
    print(f"decoders_per_rank = {decoders_per_rank}")
    split_spec = {
        f'transformer.h.{i * decoders_per_rank}': SplitPoint.BEGINNING
        for i in range(1, args.world_size)
    }

    # Create pipeline representation
    pipe = pipeline(
        gpt2,
        mb_args=(),
        mb_kwargs=mb_inputs,
        split_spec=split_spec,
    )

    assert pipe.num_stages == args.world_size, f"nstages = {pipe.num_stages} nranks = {args.world_size}"
    smod = pipe.get_stage_module(args.rank)
    print(f"Pipeline stage {args.rank} {get_number_of_params(smod) // 10 ** 6}M params")

    # ✅ 插入输出中间张量大小（MB）的 Hook
    def print_output_size_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            size_MB = output.element_size() * output.nelement() / 1e6
            print(f"[Rank {args.rank}] Output tensor size: {output.shape}, {size_MB:.2f} MB")
        elif isinstance(output, (tuple, list)):
            total_size = sum(o.element_size() * o.nelement() for o in output if isinstance(o, torch.Tensor))
            print(f"[Rank {args.rank}] Output tuple size: {total_size / 1e6:.2f} MB")
        elif isinstance(output, dict):
            total_size = sum(o.element_size() * o.nelement() for o in output.values() if isinstance(o, torch.Tensor))
            print(f"[Rank {args.rank}] Output dict size: {total_size / 1e6:.2f} MB")

    smod.register_forward_hook(print_output_size_hook)
    # Create schedule runtime
    stage = pipe.build_stage(
        args.rank,
        device=args.device,
    )

    # Attach to a schedule
    schedule = ScheduleGPipe(stage, args.chunks)

    # Full batch inputs as in single-worker case
    #inputs = generate_inputs_for_model(
    #    model_class, gpt2, model_name, args.batch_size, args.device)
    inputs = generate_inputs_for_model(model_class, gpt2, model_name, args.batch_size, args.device)
    """
    # Run
    if args.rank == 0:
        schedule.step(**inputs)
        print("\n========== Inference Output ==========\n")
        print(inputs)
    else:
        out = schedule.step()
    """
    import time

    # Run with timing
    if args.rank == 0:
        timings = []
        for i in range(args.batches):
            start = time.time()
            schedule.step(**inputs)  # Only Stage 0 passes input_ids
            end = time.time()
            timings.append(end - start)

        ttft = timings[0]
        tbt = sum(timings[1:]) / max(1, len(timings) - 1)
        print(len(timings), timings)
        print("\n========== Inference Timing ==========")
        print(f"TTFT (Time to First Token): {ttft:.4f} s")
        print(f"TBT (Time Between Tokens):  {tbt/args.batch_size:.4f} s")
        print(f"Total batches: {len(timings)}, Total time: {sum(timings):.4f} s")
    else:
        for _ in range(args.batches):
            output = schedule.step()
            logits = None
            if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                logits = output[0]
            elif isinstance(output, torch.Tensor):
                logits = output

            if logits is not None and logits.dim() == 3:  # [B, S, V]
                token_ids = logits.argmax(dim=-1)  # [B, S]
                for i in range(token_ids.size(0)):
                    tokens = token_ids[i].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    print(f"[Rank {args.rank}] Sample {i}: {text}")
    dist.barrier()
    dist.destroy_process_group()
    print(f"Rank {args.rank} completes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    # Note: this specific example requires: 1) a batch size that is divisible by
    # the number of chunks; 2) the division result (i.e. chunk size) must be 1,
    # otherwise padding token must be provided too (see GPT-2's forward function)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=None)
    parser.add_argument('--n_layer', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)

    args = parser.parse_args()

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run(args)