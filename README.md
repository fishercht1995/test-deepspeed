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

ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCyAaeQmihdErY+CUcNWdziR4kEhmsz7QZOTzqzpL+qt/pLPDJqeiOSlmTuW65M5PsEOzvfC8y9QqMN1UcuAwuBJRJ2Uu9R84QbQIU5bA96VdUzkfldIEvqnOFsHwtrbPj6N+VV3Q3GkQf9F3yPrqjEBWOQKhCGK2q7ysS5dqbSTTPW0RPntnGX6RBRorJh+EOLosUcxxtena6LCvHcRyNOUE7wrsa6EHI/7Gvb4Fhzs39mR6GyCGFxVvZiTVLYKQqhJ4qL1I2o8rqkXvvwKC2Sj6b6MCLYJxP8fBATYbDZmx15oLS5XbweVQZF4qYt1cZB0vItbMOwRMLzJACLwSZrk9Uwo3ifh4WwJRgD5G/GXKre/lMrMyxsDypWavG4gDG1b4DdUTk/DhOLey+7DKrQOizL15IR66m8079ynCy14CZYMZFgG84Idtdyv8y+VlDW4YUtnwtJFNkowdicmbY5CQ055nmnk4pEhf3JBQdQB0Zh6RtQC8WfYTeuUN/+5cHrDhDXQaYYS+zTEBW5ZhKbxzIsbGHHCUz4pu3Q/uhL9n2xFF5FvKD2d4j7RJI12fEgXAaCy0TKlUdR+faZgnvARNQN8CnBZbGxwUz+TdX5G7CW81YyIa7MkZwh7XuvLtscBT5EtapNhVJo+YhxaOnezHpJTWFq/arrg6Xmy21k7Q== ubuntu@ip-172-31-64-102

source ~/miniconda3/etc/profile.d/conda.sh
conda init
source ~/.bashrc
conda activate deepspeed-inference
pip install ninja