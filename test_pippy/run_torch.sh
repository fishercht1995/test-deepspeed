#!/bin/bash

# Usage: ./run_torch.sh <NODE_RANK>
# Example: ./run_torch.sh 0

NODE_RANK=$1

torchrun \
  --nproc-per-node=1 \
  --nnodes=2 \
  --node-rank=${NODE_RANK} \
  --master-addr=44.192.252.191 \
  --master-port=29500 \
  $2