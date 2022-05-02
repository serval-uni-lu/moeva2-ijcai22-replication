# Create the environment 
conda create -n moeva2 python=3.8.8
conda activate moeva2
pip install -r requirements.txt

# Download the additional data
wget https://figshare.com/ndownloader/files/33189926?private_link=84ae808ce6999fafd192 -O moeva-data.zip
unzip moeva-data
mv ./moeva-data/data ./data
rmdir moeva-data
rm moeva-data.zip
