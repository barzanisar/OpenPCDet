conda create --name openpcdet python=3.8
conda activate openpcdet
# Install torch for cuda 11.1
sudo apt-get install libxcb-xinerama0
pip3 install waymo-open-dataset-tf-2-6-0
python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install --upgrade -r requirements.txt
python -m pip install SharedArray==3.1.0
pip install -U urllib3 requests
