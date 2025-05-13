#!/bin/bash

SCRIPTS=$(ls test_*.py | sort)

for script in $SCRIPTS; do
  echo "▶ Running $script..." | tee -a ~/results.txt
  pdsh -w 18.232.54.0 "bash -i -c 'source ~/.bashrc && conda activate deepspeed-inference && cd /home/ubuntu/test-deepspeed/test_pippy && ./run_torch.sh 0 \"$script\" >> ~/result.txt'" &
  pdsh -w 98.80.150.148 "bash -i -c 'source ~/.bashrc && conda activate deepspeed-inference && cd /home/ubuntu/test-deepspeed/test_pippy && ./run_torch.sh 1 \"$script\" >> ~/result.txt'" &
  wait
  sleep 10
done