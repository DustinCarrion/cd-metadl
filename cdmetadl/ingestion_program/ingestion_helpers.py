import os
import shutil
import yaml
from glob import glob as ls
from typing import List


def vprint(message: str, 
           verbose: bool) -> None:
    """ Print a message based on the verbose mode.

    Args:
        message (str): Message to be printed.
        verbose (bool): Verbose mode.
    """
    if verbose: 
        print(message)

 
def exist_dir(dir: str) -> bool:
    """ Check if a directory exists.

    Args:
        dir (str): Directory to be checked.

    Raises:
        NotADirectoryError: Error raised when the directory does not exist.
        
    Returns:
        bool: True if the directory exists, an error is raised otherwise.
    """
    if os.path.isdir(dir):
        return True
    raise NotADirectoryError(f"In exist_dir, directory '{dir}' not found")
        

def print_list(lst: List[str]) -> None:
    """ Print all the elements inside a list.

    Args:
        lst (List[str]): List to be printed.
    """
    for item in lst:
        print(item)
        
        
def show_dir(dir: str = ".") -> None:
    """ Shows all the files and directories inside the specified directory up 
    to the fourth level deep.

    Args:
        dir (str, optional): Source directory to be listed. Defaults to '.'.
    """
    print(f"{'='*10} Listing directory {dir} {'='*10}")
    print_list(ls(dir))
    print_list(ls(dir + '/*'))
    print_list(ls(dir + '/*/*'))
    print_list(ls(dir + '/*/*/*'))
    print_list(ls(dir + '/*/*/*/*'))
    
        
def mvdir(source: str, 
          dest: str) -> None:
    """ Move a directory to the specified destination.

    Args:
        source (str): Current directory location.
        dest (str): Directory destination.
        
    Raises:
        OSError: Error raised when the directory cannot be renamed.
    """
    if os.path.exists(source):
        try:
            os.rename(source, dest)
        except:
            raise OSError(f"In mvdir, directory '{source}' could not be moved"
                + f" to '{dest}'")
        
        
def mkdir(dir: str) -> None:
    """ Create a directory. If the directory already exists, deletes it before 
    creating it again.

    Args:
        dir (str): Directory to be created.
        
    Raises:
        OSError: Error raised when the directory cannot not be deleted or 
            created.
    """
    if os.path.exists(dir):
        try:
            shutil.rmtree(dir)
        except:
            raise OSError(f"In mkdir, directory '{dir}' could not be deleted")
    
    try:
        os.makedirs(dir)
    except:
        raise OSError(f"In mkdir, directory '{dir}' could not be created")
        
        
def load_yaml(file: str) -> dict:
    """ Loads the content of a YAML file.

    Args:
        file (str): File in YAML format.

    Raises:
        OSError: Error raised when the file cannot be opened.

    Returns:
        dict: Content of the YAML file.
    """
    try:
        return yaml.safe_load(open(file, "r"))
    except:
        raise OSError(f"In load_yaml, file '{file}' could not be opened or "
            + f"has wrong format")
        
