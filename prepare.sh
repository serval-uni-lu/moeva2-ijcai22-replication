#!/bin/bash
# Create the environment 
conda create -n moeva2 python=3.8.8 -y
conda run -n moeva2 --no-capture-output pip install -r requirements.txt
conda run -n moeva2 --no-capture-output pip install scipy==1.4.0

# Download the additional data
wget https://figshare.com/ndownloader/files/42088389?private_link=84ae808ce6999fafd192 -O moeva-data.zip
unzip moeva-data
mv ./moeva-data/data ./data
rmdir moeva-data
rm moeva-data.zip
