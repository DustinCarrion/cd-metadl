""" This is a dummy baseline. It is just supposed to check if ingestion and 
scoring are called properly.
"""

import os
import numpy as np
import pickle
from typing import Tuple

from api import MetaLearner, Learner, Predictor


class MyMetaLearner(MetaLearner):

    def __init__(self, N_ways: int, total_train_classes: int) -> None:
        super().__init__(N_ways, total_train_classes)

    def meta_fit(self, meta_train_generator, meta_valid_generator) -> Learner:
        """ Uses the meta-dataset to fit the meta-learner's parameters. A 
        meta-dataset can be an epoch (list with batches of images) or a batch 
        of few-shot learning tasks.
        
        Args:
            meta_train_generator: Function that generates the training data.
                The generated can be an episode (N-ways any-shot learning task) 
                or a batch of images with labels.
            meta_valid_generator: Function that generates the validation data.
                The generated data always come in form of any-ways any-shot 
                learning tasks.
                
        Returns:
            Learner: Resulting learner ready to be trained and evaluated on 
                new unseen tasks.
        """
        return MyLearner()


class MyLearner(Learner):

    def __init__(self):
        super().__init__()

    def fit(self, 
            dataset_train: Tuple[np.ndarray,np.ndarray,int,int]) -> Predictor:
        """ Fit the Learner to the support set of a new unseen task. 
        
        Args:
            dataset_train: Support set of a task. The data arrive in the 
                following format (X_train, y_train, n_ways, k_shots). X_train 
                is the array of labeled imaged of shape 
                (n_ways*k_shots x 128 x 128 x 3), y_train are the encoded
                labels (int) for each image in X_train, n_ways (int) are the 
                number of classes and k_shots (int) the number of examples per 
                class.
                        
        Returns:
            Predictor: The resulting predictor ready to predict unlabelled 
                query image examples from the new unseen task.
        """
        _, y_train, _, _ = dataset_train
        return MyPredictor(y_train)

    def save(self, path_to_save: str) -> None:
        """ Saves the learning object associated to the Learner. It could be 
        a neural network for example. 
        
        Args:
            path_to_save (str): Path where the Learner will be saved
        """
        
        if not os.path.isdir(path_to_save):
            raise ValueError(("The model directory provided is invalid. Please"
                + " check that its path is valid."))
        
        pickle.dump(self, open(f"{path_to_save}/learner.pickle", "wb"))
 
    def load(self, path_to_model: str) -> Learner:
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in save().
        
        Args:
            path_to_model (str): Path where the Learner is saved
            
        Returns:
            Learner: Loaded learner
        """
        if not os.path.isdir(path_to_model):
            raise ValueError(("The model directory provided is invalid. Please"
                + " check that its path is valid."))
        
        model_file = f"{path_to_model}/learner.pickle"
        if os.path.isfile(model_file):
            with open(model_file, "rb") as f:
                saved_learner = pickle.load(f)
        return saved_learner
        
    
class MyPredictor(Predictor):

    def __init__(self, labels):
        super().__init__()
        self.labels = np.unique(labels)

    def predict(self, dataset_test) -> np.ndarray:
        """ Given a dataset_test, predicts the probabilities associated to the 
        provided images.
        
        Args:
            dataset_test: Array of unlabelled image examples of shape 
                (query_size x 128 x 128 x 3).
        
        Returns:
            np.ndarray: Predicted probs for all images. The array must be of 
                shape (query_size, N_ways).
        """
        random_pred = np.random.choice(self.labels, len(dataset_test))
        random_probs = np.zeros((random_pred.size, len(self.labels)))
        random_probs[np.arange(random_pred.size), random_pred] = 1
        return random_probs

