""" This baseline is an implementation of Model-Agnostic Meta-Learning for 
Fast Adaptation of Deep Networks (MAML) introduced by C. Finn et al. 2017 
(https://arxiv.org/pdf/1703.03400). The network initilization at meta-training
time can be with:
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
from typing import Iterable, Any, Tuple, List

from network import ResNet
from helpers_maml import *

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
        self.train_tasks = 20
        self.val_tasks = 10
        self.val_after = 5
        
        # MAML parameters
        self.base_lr = 0.01
        self.grad_clip = 10
        self.second_order = False
        self.meta_batch_size = 2
        self.T = 5
        
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
        self.meta_learner = ResNet(**self.model_args).to(self.dev)
        self.weights = [p.clone().detach().to(self.dev) for p in 
            self.meta_learner.parameters()]
        for p in self.weights:
            p.requires_grad = True
        self.optimizer = self.opt_fn(self.weights, lr=self.lr)
        
        # Store gradients across tasks
        self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in 
            self.weights]
        
        # Validation-learner
        self.best_score = -float("inf")
        self.best_state = None
        self.val_learner = ResNet(**self.model_args).to(self.dev)

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
            self.optimizer.zero_grad()
            for i, task in enumerate(meta_train_generator(self.train_tasks)):
                self.meta_learner.train()
                
                # Prepare data
                num_ways = task.num_ways
                X_train, y_train, _ = task.support_set
                X_train, y_train = X_train.to(self.dev), y_train.to(self.dev)
                X_test, y_test, _ = task.query_set
                X_test, y_test = X_test.to(self.dev), y_test.to(self.dev)
                
                # Compute loss
                task_weights = [p.clone() for p in self.weights]  
                out, loss = self.compute_out_and_loss(self.meta_learner, 
                    task_weights, X_train, y_train, X_test, y_test, num_ways, 
                    True)
                
                # Propagate loss
                loss.backward()
                
                # Clip gradients
                if self.grad_clip is not None:
                    for p in self.weights:
                        if p.grad is not None:
                            p.grad = torch.clamp(p.grad, -self.grad_clip, 
                                +self.grad_clip)
                
                # Update gradient buffer
                self.grad_buffer = [self.grad_buffer[i] + self.weights[i].grad 
                    if self.weights[i].grad is not None else 
                    self.grad_buffer[i] for i in range(len(self.weights))]
                self.optimizer.zero_grad()
                
                # Optimize metalearner
                if (i + 1) % self.meta_batch_size == 0: 
                    for i, p in enumerate(self.weights):
                        p.grad = self.grad_buffer[i]
                    self.optimizer.step()
                    
                    self.grad_buffer = [torch.zeros(p.size(), device=self.dev) 
                        for p in self.weights]
                    self.optimizer.zero_grad()
                
                # Log iteration
                self.log(task, out.detach().cpu().numpy(), loss.item())
                
                if (i + 1) % self.val_after == 0:
                    self.meta_valid(meta_valid_generator)
                    
        if self.best_state is None:
            self.best_state = [p.clone().detach() for p in self.weights]
        
        maml_params = {
            "lr": self.base_lr,
            "grad_clip": self.grad_clip,
            "second_order": self.second_order,
            "T": self.T,
            "ncc": self.ncc
        }
        return MyLearner(self.model_args, self.meta_learner.state_dict(), 
            self.best_state, maml_params)
    
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
            self.val_learner.eval()
            self.val_learner.modify_out_layer(num_ways)
            
            # Prepare weights
            task_weights = [p.clone() for p in self.weights[:-2]] 
            val_weights = [p.clone().detach().to(self.dev) for p in 
                self.val_learner.parameters()]
            task_weights.extend(val_weights[-2:])
            for p in task_weights[-2:]:
                p.requires_grad = True
            
            # Evaluate learner
            out, _ = self.compute_out_and_loss(self.val_learner, task_weights, 
                X_train, y_train, X_test, y_test, num_ways, False, True)
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
            self.best_state = [p.clone().detach() for p in self.weights]

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
    
    def compute_out_and_loss(self,
                             model: nn.Module, 
                             weights: List[torch.Tensor], 
                             X_train: torch.Tensor, 
                             y_train: torch.Tensor, 
                             X_test: torch.Tensor, 
                             y_test: torch.Tensor,
                             num_classes: int,
                             training: bool, 
                             no_loss: bool = False) -> Tuple[torch.Tensor, 
                                                             torch.Tensor]:
        """ Compute the output and loss using the specified data.

        Args:
            model (nn.Module): Model to be used.
            weights (List[torch.Tensor]): Weights to be used by the model.
            X_train (torch.Tensor): Support set images.
            y_train (torch.Tensor): Support set labels.
            X_test (torch.Tensor): Query set images.
            y_test (torch.Tensor): Query set labels.
            num_classes (int): Number of classes to predict.
            training (bool): Boolean flag to control the execution context. If 
                True, keep track of the gradients, otherwise the gradients are 
                ignored.
            no_loss (bool, optional): Boolean flag to control the loss 
                computation. If True, the loss is not computed, otherwise the 
                loss is computed. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output and loss.
        """
        # Inner step
        perform_innet_step = False if self.ncc and not training else True
        if perform_innet_step:
            retain_graph = self.second_order or self.T > 1 
            for _ in range(self.T):   
                # Compute gradients
                if self.ncc:
                    grads = get_grads_ncc(model, X_train, y_train, X_test, 
                        y_test, num_classes, weights, self.second_order, 
                        retain_graph) 
                else: 
                    grads = get_grads(model, X_train, y_train, weights, 
                        self.second_order, retain_graph)
                
                # Update task weights            
                weights = update_weights(weights, grads, self.grad_clip, 
                    self.base_lr)

        # Use torch.no_grad when evaluating
        if training:
            context = self.empty_context
        else:
            context = torch.no_grad
        
        # Get and return performance on query set
        with context():
            if self.ncc:
                prototypes = process_support_set(model, weights, X_train, 
                    y_train, num_classes)
                out = process_query_set(model, weights, X_test, prototypes)
            else:
                out = model.forward_weights(X_test, weights)
            
            if no_loss:
                loss = None
            else:
                reg = num_classes * len(y_test) if self.ncc else 1
                loss = model.criterion(out, y_test) / reg
                
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
                 weights: List[torch.Tensor] = [],
                 maml_params: dict = {}) -> None:
        """ Defines the learner initialization.
        
        Args:
            model_args (dict, optional): Arguments to initialize the learner. 
                Defaults to {}.
            model_state (dict, optional): Weights to initialize the learner. 
                Defaults to {}.
            weights (List[torch.Tensor], optional): Best weights found by the 
                meta-learner. Defaults to [].
            maml_params (dict, optional): Parameters required by MAML. Defaults
                to {}.
        """
        super().__init__()
        self.model_args = model_args
        self.model_state = model_state
        self.weights = weights
        self.maml_params = maml_params

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
        
        # Adapt learner
        self.learner.modify_out_layer(n_ways)
        
        # Prepare weights
        task_weights = [p.clone() for p in self.weights[:-2]] 
        learner_weights = [p.clone().detach().to(self.dev) for p in 
            self.learner.parameters()]
        task_weights.extend(learner_weights[-2:])
        for p in task_weights[-2:]:
            p.requires_grad = True
            
        if self.ncc:
            with torch.no_grad():
                prototypes = process_support_set(self.learner, task_weights, 
                    X_train, y_train, n_ways)
        
        else:
            prototypes = None
            # Fit weights
            retain_graph = self.second_order or self.T > 1 
            for _ in range(self.T):     
                grads = get_grads(self.learner, X_train, y_train, task_weights, 
                    self.second_order, retain_graph)
                task_weights = update_weights(task_weights, grads, 
                    self.grad_clip, self.lr)
            
        return MyPredictor(self.learner, task_weights, self.dev, prototypes)

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
        with open(f"{path_to_save}/weights.pickle", "wb+") as f:
            pickle.dump(self.weights, f)
        with open(f"{path_to_save}/maml_params.pickle", "wb+") as f:
            pickle.dump(self.maml_params, f)
 
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
            self.learner.load_params(state)
            self.learner.eval()
        else:
            raise Exception(f"'{model_state_file}' not found")
        
        weights_file = f"{path_to_load}/weights.pickle"
        if os.path.isfile(weights_file):
            with open(weights_file, "rb") as f:
                self.weights = pickle.load(f)
            for p in self.weights:
                p.requires_grad = True
        else:
            raise Exception(f"'{weights_file}' not found")
        
        maml_params_file = f"{path_to_load}/maml_params.pickle"
        if os.path.isfile(maml_params_file):
            with open(maml_params_file, "rb") as f:
                self.maml_params = pickle.load(f)
            self.lr = self.maml_params["lr"]
            self.grad_clip = self.maml_params["grad_clip"]
            self.second_order = self.maml_params["second_order"]
            self.T = self.maml_params["T"]
            self.ncc = self.maml_params["ncc"]
        else:
            raise Exception(f"'{maml_params_file}' not found")
        
    
class MyPredictor(Predictor):

    def __init__(self, 
                 model: nn.Module, 
                 weights: List[torch.Tensor],
                 dev: torch.device,
                 prototypes: torch.Tensor) -> None:
        """Defines the Predictor initialization.

        Args:
            model (nn.Module): Fitted learner.
            weights (List[torch.Tensor]): Best weights for the model.
            dev (torch.device): Device where the data is located.
            prototypes (torch.Tensor): Support prototypes.
        """
        super().__init__()
        self.model = model
        self.weights = weights
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
                out = process_query_set(self.model, self.weights, X_test, 
                    self.prototypes)
            else:
                out = self.model.forward_weights(X_test, self.weights)
            probs = F.softmax(out, dim=1).cpu().numpy()
        
        return probs
