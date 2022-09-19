conda create --name openpcdet python=3.8
conda activate openpcdet
# Install torch for cuda 11.1
python -m pip --no-cache-dir install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip --no-cache-dir install --upgrade -r requirements.txt
python -m pip install SharedArray==3.1.0
pip install -U urllib3 requests
