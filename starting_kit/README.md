# Cross-Domain Meta-Learning Competition Starting Kit

---
In this document, we present how to setup the tools to use the Jupyter Notebook tutorial. Also we present an overview of the challenge to understand how to write a valid code submission.

In this `README.md` file, you can find the following things: 
* Instructions to setup the environment to make the jupyter notebook ready to use.
* An overview of the competition workflow.  

In the **Jupyter Notebook** `tutorial.ipynb` you will learn the following things: 
* The format in which the data arrive to the meta-learning algorithm.
* Familiarize with the challenge API and more specifically how to organize your code to write a valid submission.
---

**Outline**
- [Cross-Domain Meta-Learning Competition Starting Kit](#cross-domain-meta-learning-competition-starting-kit)
  - [* Familiarize with the challenge API and more specifically how to organize your code to write a valid submission.](#-familiarize-with-the-challenge-api-and-more-specifically-how-to-organize-your-code-to-write-a-valid-submission)
  - [Setup](#setup)
    - [Download the starting kit](#download-the-starting-kit)
    - [Set up the environment with Anaconda](#set-up-the-environment-with-anaconda)
    - [Update the starting kit](#update-the-starting-kit)
    - [Public dataset](#public-dataset)
  - [Understand how a submission is evaluated](#understand-how-a-submission-is-evaluated)
  - [Prepare a ZIP file for submission on CodaLab](#prepare-a-zip-file-for-submission-on-codalab)
  - [Troubleshooting](#troubleshooting)
  - [Report bugs and create issues](#report-bugs-and-create-issues)
  - [Contact us](#contact-us)

## Setup

### Download the starting kit
You should clone the whole **cd-metadl** repository first by running the following command in the empty root directory of your project :
```
git clone https://github.com/DustinCarrion/cd-metadl.git
```
We provide detailed instructions on how to install the necessary dependencies:
- [via Conda environment](#set-up-the-environment-with-anaconda)

### Set up the environment with Anaconda

**Note** : We assume that you have already installed anaconda on your device. If it's not the case, please check out the right installation guide for your machine in the following link : [conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

A script `quick_start.sh` is available and allows to quickly set up a conda environment with all the required modules/packages installed. 

Make sure you have cloned the cd-metadl repository beforehand. 

Your root directory should look like the following: 
```
<root_directory>
|   cd-metadl
```

Then, set your current working directory to be in cd-metadl's starting kit folder using the following command:
```bash
cd cd-metadl/starting_kit/
```

Then you can run the `quick_start.sh` script:
```bash
bash quick_start.sh
```
This script creates a Python 3.8.3 conda environment named **cdml**, install all packages/modules required, and notebook.
**Note**: During the execution, the terminal will ask you to confirm the installation of packages, make sure you accept.

Once everything is installed, you can now activate your environment with this command: 
```bash
conda activate cdml
```
And launch Jupyter notebook the following way: 
```bash
jupyter-notebook
```
You will access the Jupyter menu, click on `tutorial.ipynb` and you are all set.

### Update the starting kit

As new features and possible bug fixes will be constantly added to this starting kit, 
you are invited to get the latest updates before each usage by running:

```
cd <path_to_local_cd-metadl>
git pull
```

If you forked the repository, here is how you update it: [syncing your fork](https://help.github.com/en/articles/syncing-a-fork)

### Public dataset
We will provide 10 public datasets for participants. They can use it to:
- Explore data.
- Do local test of their own algorithm.

**The link to download the public data will be available here.**

## Understand how a submission is evaluated 
First let's describe what scripts a partcipant should write to create a submission. They need to create the following files: 
- **api.py** (Mandatory): Corresponds to the API that we provide and you have to overwrite in **model.py**.
- **model.py** (Mandatory): Contains the meta-learning algorithm procedure dispatched into the appropriate classes.
- **metadata** (Mandatory): It is just a file for the competition server to work properly, you simply add it to your folder without worrying about it.
- **config.yaml** (Optionnal) : This file allows participants to meta-fit their algorithm on data with a specific **configuration**. Examples are provided in the `tutorial.ipynb`.
* **<any_file.py>** (Optionnal) : Sometimes you would need to create a specfic architecture of a neural net or any helper function for your meta-learning procedure. You can include all the files you'd like but make sure you import them correctly in **model.py** as it is the only script executed.

An example of a submission using these files is described in the provided Jupyter notebook `tutorial.ipynb`.

The following figure explains the evaluation procedure of the challenge.

![Evaluation Flow Chart](evaluation-flow-chart.png "Evaluation process of the challenge")

## Prepare a ZIP file for submission on CodaLab
Zip the contents of `baselines/random` (or any folder containing your `model.py` file) without the directory structure:
```bash
cd ../baselines/random
zip -r mysubmission.zip *
```
**Note** : The command above assumes your current working directory is `starting_kit`.

Then, the generated zip can be uploaded to the competition page on CodaLab platform.

**Tip**: One could run the following command to check the content of a zipped submission folder.
```bash
unzip -l mysubmission.zip
```
## Troubleshooting
- It is highly recommended to use the previous guidelines to prepare a zip file submission instead of simply compressing the code folder in the *Finder* (for MAC users).
- Make sure your submission always writes a file in the Learner's `save()` method. Otherwise, the submission will fail and CodaLab will return an error during the **scoring** phase.
- Remember that the `run.py` script combines the run of the ingestion and scoring process in one command. Thus, if you have an error it can be from the ingestion or scoring process. 

## Report bugs and create issues 

If you run into bugs or issues when using this starting kit, please create issues on the [*Issues* page](https://github.com/DustinCarrion/cd-metadl/issues) of this repo. 

## Contact us 
If you have any questions, please contact us via: <metalearningchallenge@googlegroups.com>

