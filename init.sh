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