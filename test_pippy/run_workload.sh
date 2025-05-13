#!/bin/bash

# List of script files
SCRIPTS=$(ls test_*.py | sort)

for script in $SCRIPTS; do
  echo "▶ Running $script..." | tee -a ~/results.txt
  pdsh -w 44.192.252.191 "conda activate deepspeed-inference && cd /home/ubuntu/test-deepspeed/test_pippy && ./run_torch.sh 0 $script" >> ~/result.txt &
  pdsh -w 3.91.112.200   "conda activate deepspeed-inference && cd /home/ubuntu/test-deepspeed/test_pippy && ./run_torch.sh 1 $script" >> ~/result.txt &
  wait
  sleep 10
done