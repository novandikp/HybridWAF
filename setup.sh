#!/bin/bash

# Add deadsnakes PPA (Personal Package Archive) for Python 3.9
sudo add-apt-repository ppa:deadsnakes/ppa

# Update package lists
sudo apt update

# Install Python 3.9
sudo apt install -y python3.9

# Install python 3.9 lib dev
sudo apt install -y libpython3.9-dev

# install pip
sudo apt install -y python3-pip

# Install all the required packages
python3.9 -m pip install -r requirements.txt
