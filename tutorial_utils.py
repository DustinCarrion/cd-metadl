import os
from sys import exit
from collections import Counter
from contextlib import closing
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from cdmetadl.helpers.ingestion_helpers import cycle
from cdmetadl.helpers.general_helpers import prepare_datasets_information
from cdmetadl.ingestion.image_dataset import create_datasets, ImageDataset
from cdmetadl.ingestion.data_generator import CompetitionDataLoader
from typing import Iterator, Any, Tuple


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
                            
        
def print_generator_info(generator: Iterator[Any], 
                         num_classes: int = None) -> None:
    """ Prints the information of a data generator.

    Args:
        generator (Iterator[Any]): Generator from which the information should 
            be printed.
        num_classes (int, optional): Total number of classes in the case of 
            batch generator. Defaults to None.
    """
    if generator is None:
        print("Since no validation_datasets were provided, the "
            + "meta_valid_generator is None. Therefore, be careful and avoid "
            + "iterating over it")
        return
        
    generated_element = next(generator(1))
    if type(generated_element) == list:
        print("\nThe batch object is organized in the following way:\n" 
            + "Example of Batch (b):\n"
            + "\t- b[0]: torch.Tensor (images)\n"
            + "\t- b[1]: torch.Tensor (labels)")
        
        print("\nThe tensor with the images has the following shape: "
            + f"{generated_element[0].shape} ([batch_size, image channels, "
            + "image height, image width])")
        print(f"The tensor with the labels has the following shape: "
            + f"{generated_element[1].shape} ([batch_size])")
        print(f"\nThere is a total of {num_classes} classes in the "
            + "concatenated dataset. Thus, the batches can contain images from"
            +" all these classes.")
        
        print("\nNote: In this competition, image channels is always 3 and "
            + "image height = image width = 128")
        print(f"\n{'*'*70}\n")
    else:
        print("\nThe task object is organized in the following way:\n" 
            + "Example of Task (t):\n"
            + f"\t- t.num_ways: int = {generated_element.num_ways}\n"
            + f"\t- t.num_shots: int = {generated_element.num_shots}\n"
            + "\t- t.support_set: Tuple[torch.Tensor, torch.Tensor, "
            + "torch.Tensor] (images, encoded labels, original labels)\n"
            + "\t- t.query_set: Tuple[torch.Tensor, torch.Tensor, "
            + "torch.Tensor] (images, encoded labels, original labels)\n"
            + f"\t- t.dataset: str = {generated_element.dataset}")
        
        print("\nThe tensor with the support set images has the following "
            + f"shape: {generated_element.support_set[0].shape} ([num_ways*"
            + "num_shots, image channels, image height, image width])")
        print(f"The support set encoded labels are: "
            + f"{generated_element.support_set[1].unique()} and the shape is: "
            + f"{generated_element.support_set[1].shape} ([num_ways*num_shots]"
            + ")")
        print(f"The support set original labels are: "
            + f"{generated_element.support_set[2].unique()} and the shape is: "
            + f"{generated_element.support_set[2].shape} ([num_ways*num_shots]"
            + ")")
        
        print("\nThe tensor with the query set images has the following shape:"
            + f" {generated_element.query_set[0].shape} ([num_ways*"
            + "query_images_per_class, image channels, image height, image "
            + "width])")
        print(f"The query set encoded labels are: "
            + f"{generated_element.query_set[1].unique()} and the shape is: "
            + f"{generated_element.query_set[1].shape} ([num_ways*"
            + "query_images_per_class])")
        print(f"The query set original labels are: "
            + f"{generated_element.query_set[2].unique()} and the shape is: "
            + f"{generated_element.query_set[2].shape} ([num_ways*"
            + "query_images_per_class])")
        
        print("\nNote: In this competition, image channels is always 3 and "
            + "image height = image width = 128")
        print(f"\n{'*'*70}\n")
        

def initialize_generators(user_config: dict, 
                          data_dir: str) -> Tuple[Iterator[Any],Iterator[Any]]:
    """ Initializes the meta-train and meta-valid generator based on the given 
    configuration.

    Args:
        user_config (dict): Configuration defined by the user.
        data_dir (str): Path to the data directory.

    Returns:
        Tuple[Iterator[Any], Iterator[Any]]: Initialized meta-train and 
            meta-valid generators. Note: the meta-valid generator can be None.
    """
    # Define the configuration for the generators
    train_data_format = "task"
    batch_size = 16
    validation_datasets = None
    
    train_generator_config = {
        "N": 5,
        "min_N": None,
        "max_N": None,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20
    }
 
    valid_generator_config = {
        "N": None,
        "min_N": 2,
        "max_N": 20,
        "k": None,
        "min_k": 1,
        "max_k": 20,
        "query_images_per_class": 20
    }
    
    if "train_data_format" in user_config:
        train_data_format = user_config["train_data_format"]
    if "batch_size" in user_config:
        batch_size = user_config["batch_size"]
        if batch_size is not None and batch_size < 1:
            print(f"Batch_size cannot be less than 1. Received: {batch_size}")
            exit(1)
    if "validation_datasets" in user_config:
        validation_datasets = user_config["validation_datasets"]
        if validation_datasets is not None and validation_datasets > 4:
            print("When tested locally validation_datasets cannot be greater "
                + f"than 4. Received: {validation_datasets}")
            exit(1)
    if "train_config" in user_config:
        train_generator_config.update(user_config["train_config"])
    if "valid_config" in user_config:
        valid_generator_config.update(user_config["valid_config"])
        
    train_generator_config["min_N"] = None
    train_generator_config["max_N"] = None

    (train_datasets_info, valid_datasets_info, 
     _) = prepare_datasets_information(data_dir, validation_datasets, 93)
    
    # Initialize genetators
    
    # Train generator
    num_train_classes = train_generator_config["N"]
    if train_data_format == "task":
        train_datasets = create_datasets(train_datasets_info)
        train_loader = CompetitionDataLoader(datasets=train_datasets, 
            episodes_config=train_generator_config, seed=93)
        meta_train_generator = train_loader.generator
    else:
        g = torch.Generator()
        g.manual_seed(93)
        train_dataset = ImageDataset(train_datasets_info)
        meta_train_generator = lambda batches: iter(cycle(batches, 
            DataLoader(dataset=train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=2, generator=g)))
        num_train_classes = len(train_dataset.idx_per_label)

    # Valid generator
    if len(valid_datasets_info) > 0:
        valid_datasets = create_datasets(valid_datasets_info)
        valid_loader = CompetitionDataLoader(datasets=valid_datasets, 
            episodes_config=valid_generator_config, seed=93)
        meta_valid_generator = valid_loader.generator
    else:
        meta_valid_generator = None

    print(f"{'*'*22} Meta-train generator info {'*'*21}")
    print_generator_info(meta_train_generator, num_train_classes)

    print(f"\n{'*'*22} Meta-valid generator info {'*'*21}")
    print_generator_info(meta_valid_generator)

    return meta_train_generator, meta_valid_generator


def plot_task(support_images: torch.Tensor, 
              support_labels: torch.Tensor, 
              query_images: torch.Tensor,
              query_labels: torch.Tensor, 
              size_multiplier: float = 2, 
              max_imgs_per_col: int = 10,
              max_imgs_per_row: int = 10) -> None:
    """ Plots the content of a task. Tasks are composed of a support set 
    (training set) and a query set (test set). 
    
    Args:
        support_images (Tensor): Images in the support set, they have a 
            shape of [support_set_size x channels x height x width].
        support_labels (Tensor): Labels in the support set, they have a 
            shape of [support_set_size]. 
        query_images (Tensor): Images in the query set, they have a 
            shape of [query_set_size x channels x height x width].
        query_labels (Tensor): Labels in the query set, they have a 
            shape of [query_set_size]. 
        size_multiplier (float, optional): Dilate or shrink the size of 
            displayed images. Defaults to 2.
        max_imgs_per_col (int, optional): Number of images in a column. 
            Defaults to 10.
        max_imgs_per_row (int, optional): Number of images in a row. Defaults 
            to 10.
    """
    support_images = np.moveaxis(support_images.numpy(), 1, -1)
    support_labels = support_labels.numpy()
    query_images = np.moveaxis(query_images.numpy(), 1, -1)
    query_labels = query_labels.numpy()

    for name, images, class_ids in zip(("Support", "Query"),
                                     (support_images, query_images),
                                     (support_labels, query_labels)):
        n_samples_per_class = Counter(class_ids)
        n_samples_per_class = {k: min(v, max_imgs_per_col) 
            for k, v in n_samples_per_class.items()}
        id_plot_index_map = {k: i for i, k
            in enumerate(n_samples_per_class.keys())}
        num_classes = min(max_imgs_per_row, len(n_samples_per_class.keys()))
        max_n_sample = max(n_samples_per_class.values())
        figwidth = max_n_sample
        figheight = num_classes
        figsize = (figheight * size_multiplier, figwidth * size_multiplier)
        fig, axarr = plt.subplots(figwidth, figheight, figsize=figsize)
        fig.suptitle(f"{name} Set", size='15')
        fig.tight_layout(pad=3, w_pad=0.1, h_pad=0.1)
        reverse_id_map = {v: k for k, v in id_plot_index_map.items()}
        for i, ax in enumerate(axarr.flat):
            ax.patch.set_alpha(0)
            # Print the class ids, this is needed since, we want to set the x 
            # axis even there is no picture.
            ax.set(xlabel=reverse_id_map[i % figheight], xticks=[], yticks=[])
            ax.label_outer()
        for image, class_id in zip(images, class_ids):
            # First decrement by one to find last spot for the class id.
            n_samples_per_class[class_id] -= 1
            # If class column is filled or not represented: pass.
            if (n_samples_per_class[class_id] < 0 or
                id_plot_index_map[class_id] >= max_imgs_per_row):
                continue
            # If width or height is 1, then axarr is a vector.
            if axarr.ndim == 1:
                ax = axarr[n_samples_per_class[class_id] 
                    if figheight == 1 else id_plot_index_map[class_id]]
            else:
                ax = axarr[n_samples_per_class[class_id], 
                    id_plot_index_map[class_id]]
            ax.imshow(image)
        plt.show()

        
def plot_batch(images: torch.Tensor, 
               labels: torch.Tensor, 
               size_multiplier: int = 1) -> None:
    """ Plot the images in a batch.

    Args:
        images (Tensor): Images inside the batch, they have a shape of 
            [batch_size x channels x height x width].
        labels (Tensor): Labels inside the batch, they have a shape of
            [batch_size].
        size_multiplier (int, optional): Dilate or shrink the size of 
            displayed images. Defaults to 1.
    """
    images = np.moveaxis(images.numpy(), 1, -1)
    labels = labels.numpy()

    num_examples = len(labels)
    figwidth = np.ceil(np.sqrt(num_examples)).astype('int32')
    figheight = num_examples // figwidth
    figsize = (figwidth * size_multiplier, (figheight + 2.5) * size_multiplier)
    _, axarr = plt.subplots(figwidth, figheight, dpi=150, figsize=figsize)

    for i, ax in enumerate(axarr.transpose().ravel()):
        ax.imshow(images[i])
        ax.set(xlabel=str(labels[i]), xticks=[], yticks=[])
    
    plt.show()


def plot_data(data: Any, 
              idx: int) -> None:
    """ Plots any type of data: batch or task.

    Args:
        data (Any): Data to be plotted.
        idx (int): Index of the data.
    """
    if type(data) == list:
        print(f"\n\nBatch {idx+1}")
        images, labels = data
        plot_batch(images, labels)
    else:
        print(f"\n\nTask {idx+1} from Dataset {data.dataset}")
        print(f"# Ways: {data.num_ways}")
        print(f"# Shots: {data.num_shots}")
        plot_task(support_images=data.support_set[0], 
                  support_labels=data.support_set[1],
                  query_images=data.query_set[0], 
                  query_labels=data.query_set[1])
