import os
from contextlib import closing
from zipfile import ZipFile, ZIP_DEFLATED


def display(path_to_file: str) -> None:
    """ Displays the content of the specified file.

    Args:
        path_to_file (str): Path to the file to be displayed.
    """
    assert os.path.isfile(path_to_file)
    with open(path_to_file, "r") as f:
        print("".join(f.readlines()))
        
        
def zipdir(archivename: str, 
           basedir: str) -> None:
    """ Zip directory, from J.F. Sebastian http://stackoverflow.com/

    Args:
        archivename (str): Name for the zip file.
        basedir (str): Directory where the submission code is located.
    """
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, _, files in os.walk(basedir):
            for fn in files:
                if not fn.endswith(".zip"):
                    absfn = os.path.join(root, fn)
                    zfn = absfn[len(basedir):] 
                    assert absfn[:len(basedir)] == basedir
                    if zfn[0] == os.sep:
                        zfn = zfn[1:]
                    z.write(absfn, zfn)
               
                    
def download_public_data():
    """ Downloads the Public Data for this competition.
    """
    try:
        import requests
    except:
        os.system("pip install requests")
        import requests
    
    current_files = os.listdir()
    if "public_data" not in current_files:
        if "public_data.zip" not in current_files:
            print("Start of download, please wait")
            res = requests.get(url = "https://codalab.lisn.upsaclay.fr/my/"+
                "datasets/download/3613416d-a8d7-4bdb-be4b-7106719053f1")
            open("public_data.zip", "wb").write(res.content)
            print("Download completed")
        print("Unzipping Public Data, please wait")
        with ZipFile("public_data.zip", "r") as zip_ref:
            zip_ref.extractall()
    print(f"The Public Data is ready")
    
    
def verify_public_data():
    current_files = os.listdir()
    if "public_data" not in current_files:
        raise Exception("\nERROR: public_data/ folder not found, please follow the process described in section 2.1 Public Data.\n")
    if len(os.listdir("public_data")) != 11: 
        raise Exception("\nERROR: public_data/ folder does not have all the necessary files, please follow the process described in section 2.1 Public Data.\n")

    imgs_per_dataset = {
        "BCT": 1320,
        "BRD": 12600,
        "CRS": 7840,
        "FLW": 4080,
        "MD_MIX": 28240,
        "PLK": 3440,
        "PLT_VIL": 1520,
        "RESISC": 1800,
        "SPT": 2920,
        "TEX": 2560
    }

    for dataset in os.listdir("public_data"):
        if dataset == "info":
            if "meta_splits.txt" not in os.listdir("public_data/info"):
                raise Exception("\nERROR: meta_splits.txt not found in public_data/info/, please follow the process described in section 2.1 Public Data.\n")
        else:
            if len(os.listdir(f"public_data/{dataset}")) != 3:
                raise Exception(f"\nERROR: public_data/{dataset}/ folder does not have all the necessary files, please follow the process described in section 2.1 Public Data.\n")
            if len(os.listdir(f"public_data/{dataset}/images")) != imgs_per_dataset[dataset]:
                raise Exception(f"\nERROR: public_data/{dataset}/images/ folder does not have all the necessary images, please follow the process described in section 2.1 Public Data.\n")

    print("Your Public Data is correct, you can continue with the tutorial")
    

