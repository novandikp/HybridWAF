#!/bin/bash

# Add deadsnakes PPA (Personal Package Archive) for Python 3.9
sudo -E add-apt-repository -y 'ppa:deadsnakes/ppa'

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


cd ./datasets/ecml

wget https://www.dropbox.com/scl/fi/kp9q56vyzmce6gqa34bd0/learning_dataset.xml?rlkey=ozsvjb9fqrx4qq797qc48yfea&st=uxix0sxy&dl=1 -O learning_dataset.xml

cd ../formatted

wget https://www.dropbox.com/scl/fi/hgetyflqk9fsii6p2l31m/ecml.json?rlkey=kgd5lumu969u8q6ggmmr8lojo&st=l24pwm3u&dl=1 -O ecml.json



