# Set up the environment with Anaconda

We assume that you have already installed Anaconda on your device. If it's not the case, please check out the right installation guide for your machine in the following link: [Anaconda installation guide](https://docs.anaconda.com/anaconda/install/).

Once you have Anaconda installed, you can use the following commands to set up your environment:

```bash
conda create -n cd-metadl python=3.8.10
conda activate cd-metadl
```

Now you have to make sure that you are in the root directory of this repository (*i.e*, `cd-metadl/`) and execute the following commands:

```bash
pip install -r requirements.txt
conda install -c conda-forge jupyterlab
```

At this point you should have your environment correctly configured. Therefore, to use it, you need to activate your environment with this command (if it is not already activated): 
```bash
conda activate cd-metadl
```

Then, you should launch Jupyter notebook the following way: 
```bash
jupyter-notebook
```

You will access the Jupyter menu, click on `tutorial.ipynb` and you are all set.