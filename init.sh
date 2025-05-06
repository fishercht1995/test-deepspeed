sudo apt update -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda init
source ~/.bashrc
conda env create -f env.yml
conda activate deepspeed-inference
pip install deepspeed
conda install transformers
rm Miniconda3-latest-Linux-x86_64.sh
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
sudo apt install pdsh -y
pip install ninja