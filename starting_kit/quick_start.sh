# Global env variables
ENV_NAME=cdml

# Create conda env
conda create -n $ENV_NAME python=3.8.10
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

cd ../
# Donwload public data
wget https://codalab.lisn.upsaclay.fr/my/datasets/download/3613416d-a8d7-4bdb-be4b-7106719053f1 -O public_data.zip
unzip public_data.zip

# Install packages in cd-metadl repo
pip install -r requirements.txt
pip install -e .

# Install notebook
conda install -c conda-forge jupyterlab
pip install jupyterlab

# tutorial.ipynb ready to use
