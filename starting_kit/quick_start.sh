# Global env variables
ENV_NAME=cdml

# Create conda env
conda create -n $ENV_NAME python=3.8.3
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install packages in cd-metadl repo
cd ../
pip install -r requirements.txt
pip install -e .

# Install notebook
conda install -c conda-forge jupyterlab
pip install jupyterlab

# tutorial.ipynb ready to use
