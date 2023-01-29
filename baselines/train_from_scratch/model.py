""" This is a basic baseline that does not perform any meta-training. During 
meta-testing it initializes the model with the weights specified by the 
MetaLearner which can be:
1. Random weights (Meta-learning league)
2. Pretrained weights (Free-style league)
"""
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Any, Tuple

from network import ResNet

from api import MetaLearner, Learner, Predictor

# --------------- MANDATORY ---------------
SEED = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)    
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# -----------------------------------------


class MyMetaLearner(MetaLearner):

    def __init__(self, 
                 train_classes: int, 
                 total_classes: int,
                 logger: Any) -> None:
        """ Defines the meta-learning algorithm's parameters. For example, one 
        has to define what would be the meta-learner's architecture. 
        
        Args:
            train_classes (int): Total number of classes that can be seen 
                during meta-training. If the data format during training is 
                'task', then this parameter corresponds to the number of ways, 
                while if the data format is 'batch', this parameter corresponds 
                to the total number of classes across all training datasets.
            total_classes (int): Total number of classes across all training 
                datasets. If the data format during training is 'batch' this 
                parameter is exactly the same as train_classes.
            logger (Logger): Logger that you can use during meta-learning 
                (HIGHLY RECOMMENDED). You can use it after each meta-train or 
                meta-validation iteration as follows: 
                    self.log(data, predictions, loss, meta_train)
                - data (task or batch): It is the data used in the current 
                    iteration.
                - predictions (np.ndarray): Predictions associated to each test 
                    example in the specified data. It can be the raw logits 
                    matrix (the logits are the unnormalized final scores of 
                    your model), a probability matrix, or the predicted labels.
                - loss (float, optional): Loss of the current iteration. 
                    Defaults to None.
                - meta_train (bool, optional): Boolean flag to control if the 
                    current iteration belongs to meta-training. Defaults to 
                    True.
        """
        # Note: the super().__init__() will set the following attributes:
        # - self.train_classes (int)
        # - self.total_classes (int)
        # - self.log (function) See the above description for details
        super().__init__(train_classes, total_classes, logger)
        
        self.dev = self.get_device()
        self.model_args = {
            "num_classes": 5, 
            "dev": self.dev, 
            "num_blocks": 18,
            "pretrained": False 
        }
        self.meta_learner = ResNet(**self.model_args).to(self.dev)

    def meta_fit(self, 
                 meta_train_generator: Iterable[Any], 
                 meta_valid_generator: Iterable[Any]) -> Learner:
        """ Uses the generators to tune the meta-learner's parameters. The 
        meta-training generator generates either few-shot learning tasks or 
        batches of images, while the meta-valid generator always generates 
        few-shot learning tasks.
        
        Args:
            meta_train_generator (Iterable[Any]): Function that generates the 
                training data. The generated can be a N-way k-shot task or a 
                batch of images with labels.
            meta_valid_generator (Iterable[Task]): Function that generates the 
                validation data. The generated data always come in form of 
                N-way k-shot tasks.
                
        Returns:
            Learner: Resulting learner ready to be trained and evaluated on new
                unseen tasks.
        """
        
        return MyLearner(self.model_args, self.meta_learner.state_dict())

    def get_device(self) -> torch.device:
        """ Get the current device, it can be CPU or GPU.

        Returns:
            torch.device: Available device.
        """
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    

class MyLearner(Learner):

    def __init__(self,
                 model_args: dict = {}, 
                 model_state: dict = {}) -> None:
        """ Defines the learner initialization.

        Args:
            model_args (dict, optional): Arguments to initialize the learner. 
                Defaults to {}.
            model_state (dict, optional): Best weights found with the 
                meta-learner. Defaults to {}.
        """
        super().__init__()
        self.opt_fn = torch.optim.Adam
        self.lr = 0.001
        self.T = 100
        self.batch_size = 4
        self.model_args = model_args
        self.model_state = model_state
        self.ncc = False

    def fit(self, support_set: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                               int, int]) -> Predictor:
        """ Fit the Learner to the support set of a new unseen task. 
        
        Args:
            support_set (Tuple[Tensor, Tensor, Tensor, int, int]): Support set 
                of a task. The data arrive in the following format (X_train, 
                y_train, original_y_train, n_ways, k_shots). X_train is the 
                tensor of labeled images of shape [n_ways*k_shots x 3 x 128 x 
                128], y_train is the tensor of encoded labels (Long) for each 
                image in X_train with shape of [n_ways*k_shots], 
                original_y_train is the tensor of original labels (Long) for 
                each image in X_train with shape of [n_ways*k_shots], n_ways is
                the number of classes and k_shots the number of examples per 
                class.
                        
        Returns:
            Predictor: The resulting predictor ready to predict unlabelled 
                query image examples from new unseen tasks.
        """
        X_train, y_train, _, n_ways, _ = support_set
        X_train, y_train = X_train.to(self.dev), y_train.to(self.dev)
        
        self.learner.modify_out_layer(n_ways)
        optimizer = self.opt_fn(self.learner.parameters(), lr=self.lr)

        if self.ncc:
            with torch.no_grad():
                # Compute input embeddings
                support_embeddings = self.learner(X_train, embedding=True)

                # Compute prototypes
                prototypes = torch.zeros((n_ways, support_embeddings.size(1)), 
                    device=self.dev)
                for i in range(n_ways):
                    mask = y_train == i
                    prototypes[i] = (support_embeddings[mask].sum(dim=0) / 
                        torch.sum(mask).item())
        else:
            # Sample T batches and make updates to the parameters 
            for _ in range(self.T):        
                X_batch, y_batch = self.get_batch(X_train, y_train,
                    self.batch_size)
                optimizer.zero_grad()
                out = self.learner(X_batch)
                loss = self.learner.criterion(out, y_batch)
                loss.backward()
                optimizer.step()
            prototypes = None
        
        return MyPredictor(self.learner, self.dev, prototypes)

    def save(self, path_to_save: str) -> None:
        """ Saves the learning object associated to the Learner. 
        
        Args:
            path_to_save (str): Path where the learning object will be saved.
        """
        
        if not os.path.isdir(path_to_save):
            raise ValueError(("The model directory provided is invalid. Please"
                + " check that its path is valid."))
        
        with open(f"{path_to_save}/model_args.pickle", "wb+") as f:
            pickle.dump(self.model_args, f)
        with open(f"{path_to_save}/model_state.pickle", "wb+") as f:
            pickle.dump(self.model_state, f)
 
    def load(self, path_to_load: str) -> None:
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in self.save().
        
        Args:
            path_to_load (str): Path where the Learner is saved.
        """
        if not os.path.isdir(path_to_load):
            raise ValueError(("The model directory provided is invalid. Please"
                + " check that its path is valid."))
        
        model_args_file = f"{path_to_load}/model_args.pickle"
        if os.path.isfile(model_args_file):
            with open(model_args_file, "rb") as f:
                self.model_args = pickle.load(f)
            self.dev = self.model_args["dev"]
            self.learner = ResNet(**self.model_args).to(self.dev)
        else:
            raise Exception(f"'{model_args_file}' not found")
        
        model_state_file = f"{path_to_load}/model_state.pickle"
        if os.path.isfile(model_state_file):
            with open(model_state_file, "rb") as f:
                state = pickle.load(f)
            self.learner.load_state_dict(state)
        else:
            raise Exception(f"'{model_state_file}' not found")
            
    def get_batch(self,
                  X: torch.Tensor, 
                  y: torch.Tensor, 
                  batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Get a batch of the specified size from the specified data.

        Args:
            X (Tensor): Images.
            y (Tensor): Labels.
            batch_size (int): Desired batch size.

        Returns:
            Tuple[Tensor, Tensor]: Batch of the specified size.
        """
        batch_indices = np.random.randint(0, X.size()[0], batch_size)
        X_batch, y_batch = X[batch_indices], y[batch_indices]
        return X_batch, y_batch
        
    
class MyPredictor(Predictor):

    def __init__(self, 
                 model: nn.Module, 
                 dev: torch.device,
                 prototypes: torch.Tensor) -> None:
        """ Defines the Predictor initialization.

        Args:
            model (nn.Module): Fitted learner. 
            dev (torch.device): Device where the data is located.
            prototypes (torch.device): Support prototypes.
        """
        super().__init__()
        self.model = model
        self.dev = dev
        self.prototypes = prototypes

    def predict(self, query_set: torch.Tensor) -> np.ndarray:
        """ Given a query_set, predicts the probabilities associated to the 
        provided images or the labels to the provided images.
        
        Args:
            query_set (Tensor): Tensor of unlabelled image examples of shape 
                [n_ways*query_size x 3 x 128 x 128].
        
        Returns:
            np.ndarray: It can be:
                - Raw logits matrix (the logits are the unnormalized final 
                    scores of your model). The matrix must be of shape 
                    [n_ways*query_size, n_ways]. 
                - Predicted label probabilities matrix. The matrix must be of 
                    shape [n_ways*query_size, n_ways].
                - Predicted labels. The array must be of shape 
                    [n_ways*query_size].
        """
        X_test = query_set.to(self.dev)
        with torch.no_grad():
            if self.prototypes is not None:
                # Compute input embeddings
                query_embeddings = self.model(X_test, embedding=True)

                # Create distance matrix (negative predictions)
                distance_matrix = (torch.cdist(query_embeddings.unsqueeze(0), 
                    self.prototypes.unsqueeze(0))**2).squeeze(0) 
                out = -1 * distance_matrix
            else:
                out = self.model(X_test)
            probs = F.softmax(out, dim=1).cpu().numpy()
        
        return probs
