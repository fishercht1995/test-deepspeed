#!/bin/bash

# Usage: ./run_torch.sh <NODE_RANK>
# Example: ./run_torch.sh 0

NODE_RANK=$1
SCRIPT=$2
shift 2
torchrun \
  --nproc-per-node=1 \
  --nnodes=2 \
  --node-rank=${NODE_RANK} \
  --master-addr=18.232.54.0 \
  --master-port=29500 \
  ${SCRIPT}