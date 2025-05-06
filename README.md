# test-deepspeed

vim /home/ubuntu/miniconda3/envs/deepspeed-inference/lib/python3.10/site-packages/torch/utils/cpp_extension.py

def is_ninja_available():
    try:
        import os
        os.environ['PATH'] += ':/home/ubuntu/miniconda3/envs/deepspeed-inference/bin'
        subprocess.check_output(['ninja', '--version']).decode()
        return True
    except Exception:
        return False



–
IPv4
Custom TCP
TCP
29500
0.0.0.0/0
–
–

IPv4
SSH
TCP
22
0.0.0.0/0
–
–
IPv4
Custom TCP
TCP
30000 - 65000
0.0.0.0/0


deepspeed --num_gpus 1 --num_nodes 2 --hostfile hosts.txt --master_addr 44.211.26.156 --master_port 29500 gpt-neo-1.3B.py >> ../va-va


source ~/miniconda3/etc/profile.d/conda.sh
conda init
source ~/.bashrc
conda activate deepspeed-inference
pip install ninja