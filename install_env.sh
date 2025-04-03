#!/bin/bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

sudo apt-get -y install cuda-toolkit-12-8 libopenmpi-dev libcudnn9-cuda-12 libcudnn9-dev-cuda-12 python3-full python3-pybind11


# install transformer engine
pip install -U pip
# install env
pip install -r requirements.txt

# wandb
# export WANDB_API_KEY=your_wandb_api_key
wandb login