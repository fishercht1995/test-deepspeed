#!/bin/bash

SCRIPTS=$(ls test_*.py | sort)

for script in $SCRIPTS; do
  echo "▶ Running $script..." | tee -a ~/results.txt
  pdsh -w 98.80.147.181 "bash -i -c 'source ~/.bashrc && conda activate deepspeed-inference && cd /home/ubuntu/test-deepspeed/test_pippy && ./run_torch.sh 0 \"$script\" >> ~/result.txt'" &
  pdsh -w 44.213.81.118 "bash -i -c 'source ~/.bashrc && conda activate deepspeed-inference && cd /home/ubuntu/test-deepspeed/test_pippy && ./run_torch.sh 1 \"$script\" >> ~/result.txt'" &
  wait
  sleep 10
done