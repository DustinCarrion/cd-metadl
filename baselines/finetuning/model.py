""" This baseline pre-trains a network with batches of data from the 
meta-training split and during meta-testing only fine-tunes the last layer. The 
network initilization at meta-training time can be with:
1. Random weights (Meta-learning league)
2. Pretrained weights (Free-style league)
"""
import os
import random
import pickle
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Any, Tuple

from network import ResNet
from helpers_finetuning import *

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
        
        # General data parameters
        self.should_train = True
        self.ncc = False
        self.support_size = 12
        self.train_batches = 20
        self.val_tasks = 10
        self.val_after = 5
        
        # General model parameters
        self.dev = self.get_device()
        self.opt_fn = torch.optim.Adam
        self.model_args = {
            "num_classes": self.train_classes, 
            "dev": self.dev, 
            "num_blocks": 18, 
            "pretrained": False 
        }
        
        # Meta-learner
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.meta_learner = ResNet(**self.model_args).to(self.dev)
        self.meta_learner.train()
        self.optimizer = self.opt_fn(self.meta_learner.parameters(), 
            lr=self.lr, betas=(self.beta1, self.beta2))
        
        # Initialize running prototypes
        if self.ncc: self.init_prototypes()
        
        # Validation-learner
        self.best_score = -float("inf")
        self.best_state = None
        self.val_lr = 0.001
        self.T = 100
        self.val_batch_size = 4
        self.val_learner = ResNet(**self.model_args).to(self.dev)
        self.val_learner.load_state_dict(self.meta_learner.state_dict())
        self.val_learner.eval()

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
        if self.should_train:
            for i, batch in enumerate(meta_train_generator(self.train_batches)):
                # Prepare data
                X_train, y_train = batch
                X_train = X_train.to(self.dev)
                y_train = y_train.view(-1).to(self.dev)
                
                if self.ncc:
                    # Create prototypes for each class
                    if 0 in self.running_lenght:
                        self.update_prototypes(X_train, y_train)

                    # Optimize metalearner
                    else:                        
                        # Prepare data as a task
                        (X_train, y_train, 
                        X_test, y_test) = self.batch_to_task(X_train, y_train, 
                            self.support_size)

                        self.optimizer.zero_grad()
                        out, loss = self.optimize_ncc(X_train, y_train, X_test, 
                            y_test, True, None)
                        loss.backward()
                        self.optimizer.step()  
                        
                        # Log iteration
                        self.log((X_test, y_test), out.detach().cpu().numpy(), 
                            loss.item())
                else:
                    # Optimize metalearner
                    out, loss = optimize_linear(self.meta_learner, 
                        self.optimizer, X_train, y_train)
                
                    # Log iteration
                    self.log(batch, out.detach().cpu().numpy(), loss.item())
                
                if (i + 1) % self.val_after == 0:
                    self.meta_valid(meta_valid_generator)
                    
        if self.best_state is None:
            self.best_state = {k : v.clone() for k, v in 
                self.meta_learner.state_dict().items()}
        
        learner_params = {
            "opt_fn": self.opt_fn,
            "lr": self.val_lr,
            "T": self.T,
            "batch_size": self.val_batch_size,
            "ncc": self.ncc
        }
        return MyLearner(self.model_args, self.best_state, learner_params)
    
    def meta_valid(self, meta_valid_generator: Iterable[Any]) -> None:
        """ Evaluate the current meta-learner with the meta-validation split 
        to select the best model.

        Args:
            meta_valid_generator (Iterable[Task]): Function that generates the 
                validation data. The generated data always come in form of 
                N-way k-shot tasks.
        """
        total_test_images = 0
        correct_predictions = 0
        for task in meta_valid_generator(self.val_tasks):
            # Prepare data
            num_ways = task.num_ways
            X_train, y_train, _ = task.support_set
            X_train, y_train = X_train.to(self.dev), y_train.to(self.dev)
            X_test, y_test, _ = task.query_set
            X_test = X_test.to(self.dev)
            
            # Adapt learner
            self.val_learner.load_params(self.meta_learner.state_dict())
            self.val_learner.freeze_layers(num_ways)
            val_optimizer = self.opt_fn(self.val_learner.parameters(), 
                self.val_lr)
            
            if self.ncc:
                # Evaluate learner
                out, _ = self.optimize_ncc(X_train, y_train, X_test, y_test, 
                    False, num_ways)
            else:
                # Optimize learner
                for _ in range(self.T):        
                    X_batch, y_batch = get_batch(X_train, y_train, 
                        self.val_batch_size)
                    optimize_linear(self.val_learner, val_optimizer, X_batch, 
                        y_batch)

                # Evaluate learner
                with torch.no_grad():
                    out = self.val_learner(X_test)
            
            preds = torch.argmax(out, dim=1).cpu().numpy()
            
            # Log iteration
            self.log(task, out.cpu().numpy(), meta_train=False)
            
            # Keep track of scores
            total_test_images += len(y_test)
            correct_predictions += np.sum(preds == y_test.numpy())

        # Check if the accuracy is better and store the new best state
        val_acc = correct_predictions / total_test_images
        if val_acc > self.best_score:
            self.best_score = val_acc
            self.best_state = {k : v.clone() for k, v in 
                self.meta_learner.state_dict().items()}

    def init_prototypes(self) -> None:
        """ Initialize the prototypes for the NCC classifier with batch 
        learning.
        """
        self.running_prototypes = torch.zeros((self.train_classes, 
            self.meta_learner.in_features), device=self.dev, 
            requires_grad=False)
        self.running_lenght = torch.zeros(self.train_classes, device=self.dev, 
            requires_grad=False)

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
    
    def update_prototypes(self, 
                          X: torch.Tensor, 
                          y: torch.Tensor,
                          grad: bool = False) -> Tuple[torch.Tensor, 
                                                       torch.Tensor]:
        """ Update the prototypes following the NCC strategy.

        Args:
            X (torch.Tensor): Images.
            y (torch.Tensor): Labels.
            grad (bool, optional): Boolean flag to indicate if the grad should
                be tracked.  
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Running prototypes and lenght 
                with gradient. It is only returned if the gradient is required.
        """
        if grad:
            context = self.empty_context
        else:
            context = torch.no_grad

        # Compute embeddings
        with context():
            current_embeddings = self.meta_learner(X, embedding=True)
            if grad:
                no_grad_embeddings = current_embeddings.clone().detach()
                grad_running_prototypes = self.running_prototypes.clone()
                grad_running_lenght = self.running_lenght.clone()
            else:
                no_grad_embeddings = current_embeddings
        
            # Update prototypes
            for i in range(self.train_classes):
                mask = y == i
                self.running_prototypes[i] += no_grad_embeddings[mask].sum(dim=0) 
                self.running_lenght[i] += torch.sum(mask).item()
                
                if grad:
                    grad_running_prototypes[i] += current_embeddings[mask].sum(dim=0) 
                    grad_running_lenght[i] += torch.sum(mask).item()
                    
        if grad:
            return grad_running_prototypes, grad_running_lenght
                
    
    def batch_to_task(self,
                      X: torch.Tensor, 
                      y: torch.Tensor, 
                      support_size: int) -> Tuple[torch.Tensor, torch.Tensor, 
                                              torch.Tensor, torch.Tensor]:
        """ Transform a batch into a task (support and query sets).

        Args:
            X (torch.Tensor): Images.
            y (torch.Tensor): Labels.
            support_size (int): Desired support size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                Support images, support labels, query images, and query labels.
        """
        # Select support and query indices
        all_indices = list(range(0, X.shape[0]))
        support_indices = np.random.choice(all_indices, size=support_size, 
            replace=False)
        query_indices = [i for i in all_indices if i not in support_indices]

        # Split batch into support and query sets
        X_train = X[support_indices, :, :, :]
        y_train = y[support_indices]
        X_test = X[query_indices, :, :, :]
        y_test = y[query_indices]

        return X_train, y_train, X_test, y_test
    
    def optimize_ncc(self,
                     X_train: torch.Tensor, 
                     y_train: torch.Tensor, 
                     X_test: torch.Tensor, 
                     y_test: torch.Tensor,
                     training: bool, 
                     num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute the output and loss using the specified data.

        Args:
            X_train (torch.Tensor): Support set images.
            y_train (torch.Tensor): Support set labels.
            X_test (torch.Tensor): Query set images.
            y_test (torch.Tensor): Query set labels.
            training (bool): Boolean flag to control the execution context. If 
                True, keep track of the gradients, otherwise the gradients are 
                ignored.
            num_classes (int): Number of classes to predict.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output and loss.
        """
        if training:
            run_prototypes, run_lenght = self.update_prototypes(X_train, 
                y_train, grad=True)
            prototypes = run_prototypes / run_lenght.unsqueeze(1)
            out = process_query_set(self.meta_learner, X_test, prototypes)
            loss = self.meta_learner.criterion(out, y_test) 
        else:
            with torch.no_grad():
                prototypes = process_support_set(self.val_learner, X_train, 
                    y_train, num_classes)
                out = process_query_set(self.val_learner, X_test, prototypes)
                loss = None
            
        return out, loss

    @contextlib.contextmanager
    def empty_context(self) -> None:
        """ Defines an empty context to avoid computing unnecessary gradients.
        """
        yield None


class MyLearner(Learner):

    def __init__(self, 
                 model_args: dict = {}, 
                 model_state: dict = {},
                 learner_params: dict = {}) -> None:
        """ Defines the learner initialization.

        Args:
            model_args (dict, optional): Arguments to initialize the learner. 
                Defaults to {}.
            model_state (dict, optional): Best weights found by the 
                meta-learner. Defaults to {}.
            learner_params (dict, optional): Parameters to be used by the 
                learner. Defaults to {}.
        """
        super().__init__()
        self.model_args = model_args
        self.model_state = model_state
        self.learner_params = learner_params

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
        
        self.learner.freeze_layers(n_ways)
        
        if self.ncc:
            with torch.no_grad():
                prototypes = process_support_set(self.learner, X_train, 
                    y_train, n_ways)
        else:
            # Optimize learner
            optimizer = self.opt_fn(self.learner.parameters(), self.lr)
            for _ in range(self.T):        
                X_batch, y_batch = get_batch(X_train, y_train, self.batch_size)
                optimize_linear(self.learner, optimizer, X_batch, y_batch)
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
        with open(f"{path_to_save}/learner_params.pickle", "wb+") as f:
            pickle.dump(self.learner_params, f)
 
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
            self.learner.eval()
            self.learner.load_state_dict(state)
        else:
            raise Exception(f"'{model_state_file}' not found")
        
        learner_params_file = f"{path_to_load}/learner_params.pickle"
        if os.path.isfile(learner_params_file):
            with open(learner_params_file, "rb") as f:
                self.learner_params = pickle.load(f)
            self.opt_fn = self.learner_params["opt_fn"]
            self.lr = self.learner_params["lr"]
            self.T = self.learner_params["T"]
            self.batch_size = self.learner_params["batch_size"]
            self.ncc = self.learner_params["ncc"]
        else:
            raise Exception(f"'{learner_params_file}' not found")
        
    
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
