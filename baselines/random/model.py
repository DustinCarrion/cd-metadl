""" This is a dummy baseline. It is just supposed to check if ingestion and 
scoring are called properly.
"""

import os
import random
import numpy as np
import pickle
from torch import Tensor
from typing import Iterable, Any, Tuple

from cdmetadl.ingestion.data_generator import Task
from cdmetadl.api.api import MetaLearner, Learner, Predictor

SEED = 98
random.seed(SEED)
np.random.seed(SEED)

class MyMetaLearner(MetaLearner):

    def __init__(self, train_classes: int) -> None:
        """ Defines the meta-learning algorithm's parameters. For example, one 
        has to define what would be the learner meta-learner's architecture. 
        
        Args:
            train_classes (int): Total number of classes that can be seen 
                during meta-training. If the data format during training is 
                'task', then this parameter corresponds to the number of ways, 
                while if the data format is 'batch', this parameter corresponds 
                to the total number of classes across all training datasets.
        """
        super().__init__(train_classes)

    def meta_fit(self, 
                 meta_train_generator: Iterable[Any], 
                 meta_valid_generator: Iterable[Task]) -> Learner:
        """ Uses the generators to tune the meta-learner's parameters. The 
        meta-training generator generates either few-shot learning tasks or 
        batches of images, while the meta-valid generator always generates 
        few-shot learning tasks.
        
        Args:
            meta_train_generator: Function that generates the training data.
                The generated can be a N-way k-shot task or a batch of images 
                with labels.
            meta_valid_generator: Function that generates the validation data.
                The generated data always come in form of N-way k-shot tasks.
                
        Returns:
            Learner: Resulting learner ready to be trained and evaluated on new
                unseen tasks.
        """
        return MyLearner()


class MyLearner(Learner):

    def __init__(self) -> None:
        """ Defines the learner initialization.
        """
        super().__init__()

    def fit(self, dataset_train: Tuple[Tensor, Tensor, int, int]) -> Predictor:
        """ Fit the Learner to the support set of a new unseen task. 
        
        Args:
            dataset_train (Tuple[Tensor, Tensor, int, int]): Support set of a 
                task. The data arrive in the following format (X_train, 
                y_train, n_ways, k_shots). X_train is the tensor of labeled 
                imaged of shape [n_ways*k_shots x 3 x 128 x 128], y_train is 
                the tensor of encoded labels (Long) for each image in X_train 
                with shape of [n_ways*k_shots], n_ways is the number of classes 
                and k_shots the number of examples per class.
                        
        Returns:
            Predictor: The resulting predictor ready to predict unlabelled 
                query image examples from new unseen tasks.
        """
        _, y_train, _, _ = dataset_train
        return MyPredictor(y_train)

    def save(self, path_to_save: str) -> None:
        """ Saves the learning object associated to the Learner. 
        
        Args:
            path_to_save (str): Path where the learning object will be saved.
        """
        
        if not os.path.isdir(path_to_save):
            raise ValueError(("The model directory provided is invalid. Please"
                + " check that its path is valid."))
        
        pickle.dump(self, open(f"{path_to_save}/learner.pickle", "wb"))
 
    def load(self, path_to_load: str) -> None:
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in self.save().
        
        Args:
            path_to_load (str): Path where the Learner is saved.
        """
        if not os.path.isdir(path_to_load):
            raise ValueError(("The model directory provided is invalid. Please"
                + " check that its path is valid."))
        
        model_file = f"{path_to_load}/learner.pickle"
        if os.path.isfile(model_file):
            with open(model_file, "rb") as f:
                saved_learner = pickle.load(f)
            self = saved_learner
        
    
class MyPredictor(Predictor):

    def __init__(self, labels: Tensor) -> None:
        """ Defines the Predictor initialization.

        Args:
            labels (Tensor): Tensor of encoded labels.
        """
        super().__init__()
        self.labels = np.unique(labels.numpy())

    def predict(self, dataset_test: Tensor) -> np.ndarray:
        """ Given a dataset_test, predicts the probabilities associated to the 
        provided images.
        
        Args:
            dataset_test (Tensor): Tensor of unlabelled image examples of shape 
                [n_ways*query_size x 3 x 128 x 128].
        
        Returns:
            np.ndarray: Predicted probs for all images. The array must be of 
                shape [n_ways*query_size, n_ways].
        """
        random_pred = np.random.choice(self.labels, len(dataset_test))
        # Mimic prediction probabilities
        random_probs = np.zeros((random_pred.size, len(self.labels)))
        random_probs[np.arange(random_pred.size), random_pred] = 1
        return random_probs
