# Functions to create meta-testing tasks for the ChaLearn Cross-domain 
# Meta-learning challenge

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, 
# AND THE WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL 
# PROPERTY RIGHTS. IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE 
# LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES 
# WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF 
# SOFTWARE, DOCUMENTS, MATERIALS, PUBLICATIONS, OR INFORMATION MADE AVAILABLE 
# FOR THE CHALLENGE. 

# Main contributors: Dustin Carrión-Ojeda, Ihsan Ullah, Isabelle Guyon, and 
# Sergio Escalera, March-August 2022

import os
import json
import numpy as np
import pandas as pd
from skimage import io
from ast import literal_eval
from sklearn.utils import shuffle
from cdmetadl.ingestion_program.ingestion_helpers import vprint
from typing import Tuple

TASK_DATA = Tuple[np.ndarray, np.ndarray] 


class Task:
    """ Class to define few-shot learning tasks.
    """
    
    def __init__(self, 
                 num_ways: int, 
                 num_shots: int, 
                 support_set: TASK_DATA, 
                 query_set: TASK_DATA, 
                 classes: np.ndarray = None,
                 dataset: str = None) -> None:
        """ 
        Args:
            num_ways (int): Number of ways (classes) for the support set. 
            num_shots (int): Number of shots (images per class) for the support 
                set.
            support_set (TASK_DATA): Support set for the few-shot task. The 
                format of the set is (np.ndarray, np.ndarray), where the first 
                array corresponds to the images and has a shape of 
                (num_ways*num_shots x 128 x 128 x 3) while the second array 
                corresponds to the labels and its shape is 
                (num_ways*num_shots, ). 
            query_set (TASK_DATA): Query set for the few-shot task. The format
                of the set is (np.ndarray, np.ndarray), where the first array 
                corresponds to the images and has a shape of 
                (query_size x 128 x 128 x 3) while the second array corresponds
                to the labels and its shape is (query_size, ). The query_size
                can vary depending on the flag to control this value. 
            classes (np.ndarray, optional): Name of the classes used in the 
                task. Defaults to None.
            dataset (str, optional): Name of the dataset used to create the 
                current task. Defaults to None.
        """
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.support_set = support_set
        self.query_set = query_set
        self.classes = classes
        self.dataset = dataset
        
        
class DataGenerator:
    """ Base class to define training and testing data generators.
    """
    
    def __init__(self, 
                 input_dir: str, 
                 min_s: int = 1,
                 max_s: int = 20,
                 fixed_query_size: bool = True,
                 query_size: int = 20,
                 verbose: bool = False) -> None:
        """
        Args:
            input_dir (str): Path to the input data directory.
            min_s (int, optional): Minimum number of shots for the generated
                tasks. Defaults to 1.
            max_s (int, optional): Maximum number of ways for the generated 
                tasks. Defaults to 20.
            fixed_query_size (bool, optional): Flag to control the size of the 
                query set. If true, query size must be specified, else, all the
                available information not used for the support set will be used
                as query set. Defaults to True.
            query_size (int, optional): Number of images for the query set. 
                Only used when fixed_query_size is True. Defaults to 20.
            verbose (bool, optional): Flag to control de verbosity of the 
                generator. Defaults to False.

        Raises:
            TypeError: Error raised when the type of an argument is invalid.
            ValueError: Error raised when the value of an argument is invalid.
            NotADirectoryError: Error raised when a path that is expected to be
                a directory is not.
        """
        # Check arguments
        if not isinstance(verbose, bool):
            raise TypeError(f"In {type(self).__name__}, only bool is valid "
                + f"argument for verbose. Received: {type(verbose)}")
 
        vprint(f"{'#'*45}\nChecking Arguments\n{'-'*45}\n", verbose)
        
        vprint(f"Checking input_dir", verbose)
        if not isinstance(input_dir, str):
            raise TypeError(f"In {type(self).__name__}, only str is valid "
                + f"argument for input_dir. Received: {type(input_dir)}")
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"In {type(self).__name__}, directory "
                + f"'{input_dir}' not found")
        
        vprint(f"Checking min_s", verbose)
        if not isinstance(min_s, int):
            raise TypeError(f"In {type(self).__name__}, only int is valid "
                + f"argument for min_s. Received: {type(min_s)}")
        if min_s < 1:
            raise ValueError(f"In {type(self).__name__}, min_s cannot be less "
                + f"than 1. Received: {min_s}")
        
        vprint(f"Checking max_s", verbose)
        if not isinstance(max_s, int):
            raise TypeError(f"In {type(self).__name__}, only int is valid "
                + f"argument for max_s. Received: {type(max_s)}")
        if max_s > 20:
            raise ValueError(f"In {type(self).__name__}, max_s cannot be "
                + f"greater than 20. Received: {max_s}")
        
        vprint(f"Checking fixed_query_size", verbose)
        if not isinstance(fixed_query_size, bool):
            raise TypeError(f"In {type(self).__name__}, only bool is valid "
                + f"argument for fixed_query_size. Received: "
                + f"{type(fixed_query_size)}")
        
        if fixed_query_size:
            vprint(f"Checking query_size", verbose)
            if not isinstance(query_size, int) and (query_size is not None):
                raise TypeError(f"In {type(self).__name__}, only int or None "
                    + f"are valid arguments for query_size. Received: "
                    + f"{type(query_size)}")
            if isinstance(query_size, int):
                if query_size < 1 or query_size > 20:
                    raise ValueError(f"In {type(self).__name__}, query_size "
                        + f"cannot be less than 1 or greater than 20. "
                        + f"Received: {query_size}")
        
        # Initialize attributes
        self._input_dir = input_dir
        self._min_s = min_s
        self._max_s = max_s
        self._fixed_query_size = fixed_query_size
        self._query_size = query_size
        self._verbose = verbose
        self._dataset_information_loaded = False
        
    def _prepare_dataset_information(self, 
                                     pool: str) -> None:
        """ Initialize all the information related to the datasets that should 
        be used.

        Args:
            pool (str): Pool of datasets that should be used. It can be 'train'
                or 'test'.

        Raises:
            Exception: Exception raised when the method is called twice.
        """
        if self._dataset_information_loaded:
            raise Exception(f"In {type(self).__name__}, "
                + f"_prepare_dataset_information cannot be called twice")
        
        self._read_datasets(pool)
        self._check_datasets()        
        self._dataset_information_loaded = True
        
    def _read_datasets(self, 
                       pool: str) -> None:
        """ Read the meta_splits.txt file and extract the path to all datasets 
        that should be used.

        Args:
            pool (str): Pool of datasets that should be used. It can be 'train'
                or 'test'.

        Raises:
            NotADirectoryError: Error raised when a path that is expected to be
                a directory is not.
            FileNotFoundError: Error raised when a path that is expected to be
                a file is not.
            OSError: Error raised when a file cannot be open.
        """
        vprint(f"{'#'*45}\nReading Datasets\n{'-'*45}\n", self._verbose)
        
        vprint(f"Checking info directory", self._verbose)
        info_dir = os.path.join(self._input_dir, "info")
        if not os.path.isdir(info_dir):
            raise NotADirectoryError(f"In {type(self).__name__}, directory "
                + f"'{info_dir}' not found")
        
        vprint(f"Checking info splits file", self._verbose)
        split_file = os.path.join(info_dir, "meta_splits.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"In {type(self).__name__}, file "
                + f"'meta_splits.txt' not found in '{info_dir}'")
        
        vprint(f"Reading splits file", self._verbose)
        try:
            with open(split_file, "r") as f:
                splits = literal_eval(f.read())
        except:
            raise OSError(f"In {type(self).__name__}, file 'meta_splits.txt' "
                + f"could not be opened or has wrong format")

        if f"meta-{pool}" not in splits:
            raise Exception(f"In {type(self).__name__}, 'meta-{pool}' not "
                + f"found in 'meta_splits.txt' file")

        vprint(f"Saving datasets folders", self._verbose)
        self._datasets = [os.path.join(self._input_dir, dataset) 
            for dataset in splits[f"meta-{pool}"]]
                
        vprint(f"\n{'-'*45}\nReading Datasets Finished Successfully\n{'#'*45}"
            + f"\n\n", self._verbose)
                
    def _check_datasets(self) -> None:
        """ Check the format of all dataset that should be used. The format can
        be found in 
        https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat.
        
        Raises:
            NotADirectoryError: Error raised when a path that is expected to be
                a directory is not.
            FileNotFoundError: Error raised when a path that is expected to be
                a file is not.
            OSError: Error raised when a file cannot be opened.
            Exception: Exception raised when specific columns are not presented
                in the formatted datasets.
        """
        FORMAT_MSG = "Make sure your datasets follow this format: " \
            + "https://github.com/ihsaan-ullah/meta-album/tree/master/"\
            + "DataFormat"
        
        vprint(f"{'#'*45}\nChecking Datasets Structure\n{'-'*45}\n", 
            self._verbose)
        self._datasets_info = dict()
        for dataset in self._datasets:
            vprint(f"Checking {dataset}", self._verbose)
            
            # Define paths
            IMAGES_PATH = os.path.join(dataset, "images")
            JSON_PATH = os.path.join(dataset, "info.json")
            CSV_PATH = os.path.join(dataset, "labels.csv")

            # Check paths
            vprint(f"\tChecking image path", self._verbose)
            if not os.path.isdir(IMAGES_PATH):
                raise NotADirectoryError(f"In {type(self).__name__}, directory"
                    + f" '{IMAGES_PATH}' not found. {FORMAT_MSG}")

            vprint(f"\tChecking json file", self._verbose)
            if not os.path.isfile(JSON_PATH):
                raise FileNotFoundError(f"In {type(self).__name__}, file "
                    + f"'info.json' not found in '{dataset}'. {FORMAT_MSG}")

            vprint(f"\tChecking CSV file", self._verbose)
            if not os.path.isfile(CSV_PATH):
                raise FileNotFoundError(f"In {type(self).__name__}, file "
                    + f"'labels.csv' not found in '{dataset}'. {FORMAT_MSG}")

            vprint(f"\tReading JSON file", self._verbose)
            try:
                # Read JSON
                with open(JSON_PATH, 'r') as f:
                    info = json.loads(f.read())
                    
                # Retrieve settings from JSON
                IMAGE_COLUMN = info["image_column_name"]
                CATEGORY_COLUMN = info["category_column_name"]
                HAS_SUPER_CATEGORY = info["has_super_categories"]
                SUPER_CATEGORY_COLUMN = info["super_category_column_name"]
            except Exception as e:
                print(e)
                raise OSError(f"In {type(self).__name__}, file 'info.json' "
                    + f"located in '{dataset}' could not be opened or has "
                    + f"wrong format. {FORMAT_MSG}")

            # Read CSV
            vprint(f"\tReading CSV file", self._verbose)    
            try:        
                data_df = pd.read_csv(CSV_PATH, encoding="utf-8")
            except:
                raise OSError(f"In {type(self).__name__}, file 'labels.csv' "
                    + f"located in '{dataset}' could not be opened or has "
                    + f"wrong format. {FORMAT_MSG}")

            # Check columns in CSV
            vprint(f"\tChecking columns in CSV file", self._verbose)    
            csv_columns = data_df.columns

            # Image
            if not IMAGE_COLUMN in csv_columns:
                raise Exception(f"In {type(self).__name__}, column "
                    + f"'{IMAGE_COLUMN}' not found in 'labels.csv' located in "
                    + f"'{dataset}'. {FORMAT_MSG}")

            # Category
            if not CATEGORY_COLUMN in csv_columns:
                raise Exception(f"In {type(self).__name__}, column "
                    + f"'{CATEGORY_COLUMN}' not found in 'labels.csv' located "
                    + f"in '{dataset}'. {FORMAT_MSG}")

            # Super Category
            if HAS_SUPER_CATEGORY:
                if not SUPER_CATEGORY_COLUMN in csv_columns:
                    raise Exception(f"In {type(self).__name__}, column "
                        + f"'{SUPER_CATEGORY_COLUMN}' not found in "
                        + f"'labels.csv' located in '{dataset}'. {FORMAT_MSG}")

            vprint(f"{dataset} has correct format\n", self._verbose)

            self._datasets_info[dataset] = (IMAGES_PATH, CSV_PATH, 
                CATEGORY_COLUMN, IMAGE_COLUMN) 

        vprint(f"\n{'-'*45}\nChecking Datasets Structure Finished Successfully"
            + f"\n{'#'*45}\n\n", self._verbose)

    def _create_task(self, 
                     dataset_info: Tuple[str, str, str, str], 
                     data_df: pd.core.frame.DataFrame,
                     num_ways: int, 
                     num_shots: int,
                     private_info: bool = True
                     ) -> Tuple[TASK_DATA, TASK_DATA, np.ndarray]:
        """ Creates the support and query sets for a few-shot learning task 
        following the specified information.

        Args:
            dataset_info (Tuple[str, str, str, str]): General dataset 
                information: (IMAGES_PATH, CSV_PATH, CATEGORY_COLUMN, 
                IMAGE_COLUMN).
            data_df (pd.core.frame.DataFrame): Dataframe with information of 
                the image file names and classes.
            num_ways (int): Number of ways (classes) for the support set.
            num_shots (int): Number of shots (images per class) for the query 
                set.
            private_info (bool, optional): Boolean flag indicating whether the 
                classes should be kept private or not. Default to True.

        Raises:
            Exception: Exception raised when the method is called before 
                loading the dataset information.

        Returns:
            Tuple[TASK_DATA, TASK_DATA, np.ndarray]: Support and query sets and 
                classes used in the task. Each set has the following format: 
                (np.ndarray, np.ndarray), where the first array corresponds to 
                the images and the second array corresponds to the labels.
        """
        if not self._dataset_information_loaded:
            raise Exception(f"In {type(self).__name__}, _create_task cannot be"
                + f" called without retrieving the dataset information")
            
        vprint(f"{'#'*45}\nCreating Few-Shot Task\n{'-'*45}\n", self._verbose)
        
        # Extract information
        IMAGES_PATH, _, CATEGORY_COLUMN, IMAGE_COLUMN = dataset_info

        vprint(f"Selecting the ways (classes)", self._verbose)
        data_df['label_cat'] = data_df[CATEGORY_COLUMN].astype('category')
        ways = np.random.choice(data_df['label_cat'].cat.categories.values, 
            num_ways, replace=False)
        data_df = data_df[data_df["label_cat"].isin(ways)].copy(deep=True)
        if private_info:
            ways = np.unique(np.asarray(data_df['label_cat'].cat.codes.values))
        data_df['label_cat'] = data_df[CATEGORY_COLUMN].astype('category')

        vprint(f"Creating the support and query sets", self._verbose)
        support_df = shuffle(data_df.groupby('label_cat', observed=True, 
            as_index=False).apply(lambda df: df.sample(n=num_shots, 
            replace=False)))
        query_df = shuffle(data_df[-data_df[IMAGE_COLUMN].isin(
            support_df[IMAGE_COLUMN])])
        if self._fixed_query_size:
            query_df = query_df.sample(n=self._query_size, replace=False)
        
        vprint(f"Extracting the information from the support and query sets", 
            self._verbose)
        y_train = np.asarray(support_df['label_cat'].cat.codes.values)
        vprint(f"Loading the support images", self._verbose)
        X_train = list()
        for image_name in support_df[IMAGE_COLUMN].values:
            file = f"{IMAGES_PATH}/{image_name}"
            img = io.imread(file)
            X_train.append(img)
        X_train = np.asarray(X_train)

        y_test = np.asarray(query_df['label_cat'].cat.codes.values)
        vprint(f"Loading the query images", self._verbose)
        X_test = list()
        for image_name in query_df[IMAGE_COLUMN].values:
            file = f"{IMAGES_PATH}/{image_name}"
            img = io.imread(file)
            X_test.append(img)
        X_test = np.asarray(X_test)
        
        vprint(f"\n{'-'*45}\nCreating Few-Shot Task Finished Successfully\n"
            + f"{'#'*45}\n\n", self._verbose)

        return (X_train, y_train), (X_test, y_test), ways


class TrainGenerator(DataGenerator):
    """ Class to define the train data generator.
    
    Usage example:
        At meta-train time:
            generator = TrainGenerator(input_dir=path)
            meta_train_generator = generator.meta_train_generator
            meta_valid_generator = generator.meta_valid_generator
    """
    
    def __init__(self,
                 input_dir: str, 
                 data_format: str = "episode",
                 train_pool_size: float = 0.75,
                 num_ways: int = 5,
                 min_s: int = 1,
                 max_s: int = 20,
                 fixed_query_size: bool = True,
                 query_size: int = 20,
                 verbose: bool = False) -> None:
        """
        Args:
            input_dir (str): Path to the input data directory.
            data_format (str, optional): Format for the training data, it can 
                be 'episode' or 'batch'. The former will produce few-shot 
                learning tasks for meta-training while the latter will produce 
                batches of data for meta-training. Defaults to 'episode'.
            train_pool_size (float, optional): Percentage of the available 
                classes that should be used to generate the training examples. 
                The remaining percentage will be kept for validation. Defaults 
                to 0.75.
            num_ways (int, optional): Number of ways (classes) for the support 
                set. Only used when data_format is 'episode'. Defaults to 5.
            min_s (int, optional): Minimum number of shots for the generated 
                tasks (training and validation). Defaults to 1.
            max_s (int, optional): Maximum number of shots for the generated 
                tasks (training and validation). Defaults to 20.
            fixed_query_size (bool, optional): Flag to control the size of the 
                query set. If true, query size must be specified, else, all the
                available information not used for the support set will be used
                as query set. Defaults to True.
            query_size (int, optional): Number of images for the query set. 
                Only used when fixed_query_size is True. Defaults to 20.
            verbose (bool, optional): Flag to control de verbosity of the 
                generator. Defaults to False.

        Raises:
            TypeError: Error raised when the type of an argument is invalid.
            ValueError: Error raised when the value of an argument is invalid.
        """
        super().__init__(input_dir = input_dir,
                         min_s = min_s,
                         max_s = max_s,
                         fixed_query_size = fixed_query_size,
                         query_size = query_size,
                         verbose = verbose)
        
        # Check arguments
        vprint(f"Checking data_format", verbose)
        if not isinstance(data_format, str):
            raise TypeError(f"In {type(self).__name__}, only str is valid "
                + f"argument for data_format. Received: {type(data_format)}")
        data_format = data_format.lower()
        if data_format not in ["episode", "batch"]:
            raise ValueError(f"In {type(self).__name__}, only 'episode' or "
                + f"'batch' are valid arguments for data_format. Received: "
                + f"{data_format}")
        
        vprint(f"Checking train_pool_size", verbose)
        if not isinstance(train_pool_size, float):
            raise TypeError(f"In {type(self).__name__}, only float is valid "
                + f"argument for train_pool_size. Received: "
                + f"{type(train_pool_size)}")
        if train_pool_size < 0.1 or train_pool_size > 0.8:
            raise ValueError(f"In {type(self).__name__}, train_pool_size "
                + f"cannot be less than 0.1 or greater than 0.8. Received: "
                + f"{train_pool_size}")
        
        vprint(f"\n{'-'*45}\nChecking Arguments Finished Successfully\n"
            + f"{'#'*45}\n\n", verbose)
        
        # Prepare datasets information
        self._prepare_dataset_information("train")
        self._train_pool_size = train_pool_size
        self._data_format = data_format
        self._create_pools()
        
        # Attributes initialization
        self.num_ways = num_ways
        self._number_of_datasets = len(self._datasets)
        self._selection_counter = {
            "train": np.ones(self._number_of_datasets),
            "validation": np.ones(self._number_of_datasets)
        } 
        
        # Assign generators
        if self._data_format == "episode":
            self.meta_train_generator = lambda episodes = 50: \
                self._generate_episodes("train", episodes)
        else:
            self.meta_train_generator = self._generate_batches
        
        self.meta_valid_generator = lambda episodes = 25: \
            self._generate_episodes("validation", episodes)
    
    def _create_pools(self) -> None:
        """ Creates the train and validation pools for all the datasets.
        """
        self._pool_info = dict()
        self.total_train_classes = 0
        self._train_data = None
        self._IMAGE_PATH = "IMAGE_PATH"
        for dataset in self._datasets:
            # Read dataset information
            IMAGES_PATH, CSV_PATH, CATEGORY_COLUMN, IMAGE_COLUMN = \
                self._datasets_info[dataset]
            data_df = pd.read_csv(CSV_PATH, encoding="utf-8")
            categories = data_df[CATEGORY_COLUMN].unique()
            
            # Randomly create train and validation pools
            TRAIN_POOL = np.random.choice(categories, 
                int(self._train_pool_size * len(categories)), replace=False)  
            VALIDATION_POOL = categories[~np.in1d(categories, TRAIN_POOL)]       
            self._pool_info[dataset] = {"train": TRAIN_POOL,
                "validation": VALIDATION_POOL}
            self.total_train_classes += len(TRAIN_POOL)
            
            # Save the train data in case of batch training
            if self._data_format == "batch":
                data_df = data_df[data_df[CATEGORY_COLUMN].isin(TRAIN_POOL)]
                data_df = data_df[[IMAGE_COLUMN, CATEGORY_COLUMN]].copy(
                    deep=True)
                data_df[self._IMAGE_PATH] = IMAGES_PATH
                if self._train_data is None:
                    self._IMAGE_COLUMN = IMAGE_COLUMN
                    self._train_data = data_df
                else:
                    self._train_data = pd.concat([self._train_data, data_df], 
                        ignore_index=True, sort=False)
        
        if self._data_format == "batch":
            self._train_data = shuffle(self._train_data)
            self._train_data["label_cat"] = self._train_data[CATEGORY_COLUMN
                ].astype("category")
                        
    def _generate_episodes(self,
                           pool: str,
                           number_of_episodes: int = 50) -> Task:
        """ Creates a generator of few-shot learning episodes. If the pool is 
        'train' the episodes are N-way any-shot while if the pool is 
        'validation' the episodes are any-way any-shot.

        Args:
            pool (str): Pool of data that should be used. It can be 'train' or 
                'validation'.
            number_of_episodes (int, optional): Number of episodes to be 
                generated. Defaults to 50.
                
        Raises:
            Exception: Exception raised when the new task cannot be generated.

        Yields:
            Task: Generated task.
        """
        for _ in range(number_of_episodes):
            # Randomly sample dataset
            dataset_selection_probability = self._selection_counter[pool] \
                / np.sum(self._selection_counter[pool])
            dataset = np.random.choice(self._datasets, 
                p=dataset_selection_probability)
            self._selection_counter[pool][np.delete(np.arange(
                self._number_of_datasets), self._datasets.index(dataset))] += 1
            
            categories = self._pool_info[dataset][pool]
            num_of_categories = len(categories)
            
            # Read dataset information
            dataset_info = self._datasets_info[dataset]
            data_df = pd.read_csv(dataset_info[1], encoding="utf-8")
            data_df = data_df[data_df[dataset_info[2]].isin(
                categories)].copy(deep=True)
                
            # Randomly sample task parameters
            if pool == "train":
                num_ways = min(self.num_ways, num_of_categories)
            else:
                max_w = min(num_of_categories, 20)
                num_ways = np.random.randint(2, max_w + 1)
            num_shots = np.random.randint(self._min_s, self._max_s + 1)
            
            # Create the support and query sets
            try:
                support_set, query_set, _ = self._create_task(dataset_info, 
                    data_df, num_ways, num_shots)
            except:
                raise Exception(f"In {type(self).__name__}, few-shot task "
                    + f"could not be generated")
            
            yield Task(num_ways, num_shots, support_set, query_set) 

    def _generate_batches(self,
                          number_of_batches: int = 10,
                          batch_size: int = 400) -> TASK_DATA:
        """ Creates a generator of batches of images. Each batch can contain 
        information from all the datasets.

        Args:
            number_of_batches (int, optional): Number of batches to be 
                generated. Defaults to 10.
            batch_size (int, optional): Number of images per batch. Defaults 
                to 400.

        Raises:
            ValueError: Error raised when the batch size is invalid.

        Yields:
            TASK_DATA: Generated batch. Each batch is composed of images and 
                labels. The first array corresponds to the images and has a 
                shape of (batch_size x 128 x 128 x 3) while the second array 
                corresponds to the labels and its shape is (batch_size, ).
        """
        for _ in range(number_of_batches):
            vprint(f"{'#'*45}\nCreating Batch\n{'-'*45}\n", self._verbose)
            try:
                batch_df = self._train_data.sample(n=batch_size, replace=False)
            except:
                raise ValueError(f"In {type(self).__name__}, batch_size "
                    + f"exceeded maximum available data "
                    + f"({len(self._train_data)}). Received: {batch_size}")
            labels = np.asarray(batch_df["label_cat"].cat.codes.values)

            vprint(f"Loading the images", self._verbose)
            data = list()
            for j in range(len(batch_df)):
                row = batch_df.iloc[j]
                file = f"{row[self._IMAGE_PATH]}/{row[self._IMAGE_COLUMN]}"
                img = io.imread(file)
                data.append(img)
            data = np.asarray(data)
            
            yield (data, labels)


class TestGenerator(DataGenerator):
    """ Class to define the test data generator.
    
    Usage example:
        At meta-test time:
            generator = TestGenerator(input_dir=path)
            meta_test_generator = generator.meta_test_generator
    """
    
    def __init__(self,
                 input_dir: str, 
                 min_w: int = 2,
                 max_w: int = 20,
                 min_s: int = 1,
                 max_s: int = 20,
                 fixed_query_size: bool = True,
                 query_size: int = 20,
                 private_info = True,
                 verbose: bool = False) -> None:
        """
        Args:
            input_dir (str): Path to the input data directory.
            min_w (int, optional): Minimum number of ways for the generated
                tasks. Defaults to 2.
            max_w (int, optional): Maximum number of ways for the generated
                tasks. Defaults to 20.
            min_s (int, optional): Minimum number of shots for the generated
                tasks. Defaults to 1.
            max_s (int, optional): Maximum number of ways for the generated 
                tasks. Defaults to 20.
            fixed_query_size (bool, optional): Flag to control the size of the 
                query set. If True, query size must be specified, else, all the
                available information not used for the support set will be used
                as query set. Defaults to True.
            query_size (int, optional): Number of images for the query set. 
                Only used when fixed_query_size is True. Defaults to 20.
            private_info (bool, optional): Flag to control the privacy of 
                the information. If True, the datasets name and classes are 
                kept private, else, the information is shown. Defaults to True.
            verbose (bool, optional): Flag to control de verbosity of the 
                generator. Defaults to False.

        Raises:
            TypeError: Error raised when the type of an argument is invalid.
            ValueError: Error raised when the value of an argument is invalid.
        """
        super().__init__(input_dir = input_dir,
                         min_s = min_s,
                         max_s = max_s,
                         fixed_query_size = fixed_query_size,
                         query_size = query_size,
                         verbose = verbose)
        
        # Check arguments
        vprint(f"Checking min_w", verbose)
        if not isinstance(min_w, int):
            raise TypeError(f"In {type(self).__name__}, only int is valid "
                + f"argument for min_w. Received: {type(min_w)}")
        if min_w < 2:
            raise ValueError(f"In {type(self).__name__}, min_w cannot be less "
                + f"than 2. Received: {min_w}")
        
        vprint(f"Checking max_w", verbose)
        if not isinstance(max_w, int):
            raise TypeError(f"In {type(self).__name__}, only int is valid "
                + f"argument for max_w. Received: {type(max_w)}")
        if max_w > 20:
            raise ValueError(f"In {type(self).__name__}, max_w cannot be "
                + f"greater than 20. Received: {max_w}")
            
        vprint(f"Checking private_info", verbose)
        if not isinstance(private_info, bool):
            raise TypeError(f"In {type(self).__name__}, only bool is valid "
                + f"argument for private_info. Received: "
                + f"{type(private_info)}")
            
        vprint(f"\n{'-'*45}\nChecking Arguments Finished Successfully"
            + f"\n{'#'*45}\n\n", verbose)
        
        # Prepare datasets information
        self._prepare_dataset_information("test")
        self._private_names = {dataset: f"Dataset {i+1}" for i, dataset in 
            enumerate(self._datasets)}
        
        # Initialize attributes
        self._min_w = min_w
        self._max_w = max_w
        self._private_info = private_info
        self._number_of_datasets = len(self._datasets)
        self._selection_counter = np.ones(self._number_of_datasets) 
        self.meta_test_generator = self._generate_episodes
        
    def _generate_episodes(self,
                           number_of_episodes: int = 500) -> Task:
        """ Creates a generator of any-way any-shot few-shot learning episodes. 

        Args:
            number_of_episodes (int, optional): Number of episodes to be 
                generated. Defaults to 500.

        Raises:
            Exception: Exception raised when the new task cannot be generated.

        Yields:
            Task: Generated task.
        """
        for _ in range(number_of_episodes):
            # Randomly sample dataset
            dataset_selection_probability = self._selection_counter \
                / np.sum(self._selection_counter)
            dataset = np.random.choice(self._datasets, 
                p=dataset_selection_probability)
            self._selection_counter[np.delete(np.arange(
                self._number_of_datasets), self._datasets.index(dataset))] += 1
            if self._private_info:
                dataset_name = self._private_names[dataset]
            else:
                dataset_name = dataset.split("\\")[-1].split("/")[-1]
            
            # Read dataset information
            dataset_info = self._datasets_info[dataset]
            data_df = pd.read_csv(dataset_info[1], encoding="utf-8")
            max_w = len(data_df[dataset_info[2]].unique())
            if self._max_w < max_w:
                max_w = self._max_w
            
            # Randomly sample task parameters
            num_ways = np.random.randint(self._min_w, max_w + 1)
            num_shots = np.random.randint(self._min_s, self._max_s + 1)
            
            # Create the support and query sets
            try:
                support_set, query_set, classes = self._create_task(
                    dataset_info, data_df, num_ways, num_shots, 
                    self._private_info)
            except:
                raise Exception(f"In {type(self).__name__}, few-shot task "
                    + f"could not be generated")
            
            yield Task(num_ways, num_shots, support_set, query_set, 
                classes, dataset_name) 
