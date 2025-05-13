import os
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Initialize distributed
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Load model config and tokenizer
config = GPT2Config()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Split GPT2 into two parts manually
class GPT2BlockSplit(nn.Module):
    def __init__(self, config, start, end):
        super().__init__()
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([GPT2Model(config).h[i] for i in range(start, end)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_embeds):
        hidden_states = input_embeds
        for block in self.blocks:
            hidden_states = block(hidden_states)[0]
        return self.ln_f(hidden_states)

# Create embedding and output layers only in rank 0
if rank == 0:
    config.n_layer = 12
    embed = nn.Embedding(config.vocab_size, config.n_embd).to(device)
    pos_embed = nn.Embedding(config.n_positions, config.n_embd).to(device)
    drop = nn.Dropout(config.embd_pdrop).to(device)
    lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False).to(device)

    block0 = GPT2BlockSplit(config, 0, config.n_layer // 2).to(device)
    block1 = GPT2BlockSplit(config, config.n_layer // 2, config.n_layer).to(device)
else:
    block0 = GPT2BlockSplit(config, 0, config.n_layer // 2).to(device)
    block1 = GPT2BlockSplit(config, config.n_layer // 2, config.n_layer).to(device)

# Prompt and prepare input
prompt = "The quick brown fox"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Run pipeline
if rank == 0:
    pos_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device)
    input_embeds = embed(input_ids) + pos_embed(pos_ids)
    hidden = drop(input_embeds)
    hidden = block0(hidden)
    dist.send(hidden.cpu(), dst=1)
    hidden_recv = torch.empty_like(hidden.cpu())
    dist.recv(hidden_recv, src=1)
    logits = lm_head(hidden_recv.to(device))
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    print("Next token:", tokenizer.decode(next_token))

elif rank == 1:
    hidden_recv = torch.empty(1, input_ids.size(1), config.n_embd)
    dist.recv(hidden_recv, src=0)
    hidden = block1(hidden_recv.to(device))
    dist.send(hidden.cpu(), dst=0)

# Cleanup
dist.barrier()
dist.destroy_process_group()