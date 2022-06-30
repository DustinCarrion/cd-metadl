""" This baseline is borrowed from MetaDelta++: Improve Generalization
of Few-shot System Through Multi-Scale Pretrained Models and Improved 
Training Strategies introduced by Chen & Guan & Wei et al. 2021.
(https://arxiv.org/abs/2102.10744)

Adopted from https://github.com/Frozenmad/MetaDelta
"""
import pickle
import time
import random

TIME_LIMIT = 4500 # time limit of the whole process in seconds
TIME_TRAIN = TIME_LIMIT - 30*60 # set aside 30min for test
t1 = time.time()

import os
import torch

try:
    import numpy as np
except:
    os.system("pip install numpy")

try:
    import cython
except:
    os.system("pip install cython")

try:
    import ot
except:
    os.system("pip install POT")

try:
    import tqdm
except:
    os.system("pip install tqdm")

try:
    import timm
except:
    os.system("pip install timm")

from utils import get_logger, timer, resize_tensor, augment, decode_label, mean
from api import MetaLearner, Learner, Predictor
from backbone import MLP, rn_timm_mix, Wrapper
from torch import optim
import torch.nn.functional as F
from typing import Iterable, Any, Tuple, List

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

LOGGER = get_logger('GLOBAL')
DEVICE = torch.device('cuda')

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
        
        self.timer = timer()
        self.timer.initialize(time.time(), TIME_TRAIN - time.time() + t1)
        self.timer.begin('load pretrained model')
        self.model = Wrapper(rn_timm_mix(True, 'swsl_resnet50', 
            0.1)).to(DEVICE)
        
        times = self.timer.end('load pretrained model')
        LOGGER.info('current model', self.model)
        LOGGER.info('load time', times, 's')
        self.dim = 2048

        # only optimize the last 2 layers
        backbone_parameters = []
        backbone_parameters.extend(self.model.set_get_trainable_parameters([3, 
            4]))
        # set learnable layers
        self.model.set_learnable_layers([3, 4])
        self.cls = MLP(self.dim, train_classes).to(DEVICE)
        self.opt = optim.Adam(
            [
                {"params": backbone_parameters},
                {"params": self.cls.parameters(), "lr": 1e-3}
            ], lr=1e-4
        )

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
        # fix the valid dataset for fair comparison
        valid_task = []
        for task in meta_valid_generator(50):
            # fixed 5-way 5-shot 5-query settings
            supp_x, supp_y = task.support_set[0], task.support_set[1]
            quer_x, quer_y = task.query_set[0], task.query_set[1]
            supp_x = supp_x[supp_y.sort()[1]]
            supp_end = supp_x.size(0)
            valid_task.append([torch.cat([resize_tensor(supp_x, 224), 
                resize_tensor(quer_x, 224)]), quer_y])

        # loop until time runs out
        total_epoch = 0

        # eval ahead
        with torch.no_grad():
            self.model.set_mode(False)
            acc_valid = 0
            for x, quer_y in valid_task:
                x = x.to(DEVICE)
                x = self.model(x)
                supp_x, quer_x = x[:supp_end], x[supp_end:]

                supp_x = supp_x.view(5, 5, supp_x.size(-1))
                logit = decode_label(supp_x, quer_x).cpu().numpy()
                acc_valid += (logit.argmax(1) == np.array(quer_y)).mean()
            acc_valid /= len(valid_task)
            LOGGER.info("epoch %2d valid mean acc %.6f" % (total_epoch,
                acc_valid))

        best_valid = acc_valid
        best_param = pickle.dumps(self.model.state_dict())

        self.cls.train()
        while self.timer.time_left() > 60 * 5:
            # train loop
            self.model.set_mode(True)
            for _ in range(5):
                total_epoch += 1
                self.opt.zero_grad()
                err = 0
                acc = 0
                for i, batch in enumerate(meta_train_generator(10)):
                    self.timer.begin('train data loading')
                    X_train, y_train = batch
                    X_train = augment(X_train)
                    X_train = resize_tensor(X_train, 224)
                    X_train = X_train.to(DEVICE)
                    y_train = y_train.view(-1).to(DEVICE)
                    self.timer.end('train data loading')

                    self.timer.begin('train forward')
                    feature = self.model(X_train)
                    logit = self.cls(feature)
                    loss = F.cross_entropy(logit, y_train) / 10.
                    self.timer.end('train forward')

                    self.timer.begin('train backward')
                    loss.backward()
                    self.timer.end('train backward')

                    err += loss.item()
                    acc += logit.argmax(1).eq(y_train).float().mean()

                backbone_parameters = []
                backbone_parameters.extend(
                    self.model.set_get_trainable_parameters([3, 4]))
                torch.nn.utils.clip_grad.clip_grad_norm_(backbone_parameters + 
                    list(self.cls.parameters()), max_norm=5.0)
                self.opt.step()
                acc /= 10
                LOGGER.info('epoch %2d error: %.6f acc %.6f | time cost - dataload: %.2f forward: %.2f backward: %.2f' % (
                    total_epoch, err, acc,
                    self.timer.query_time_by_name("train data loading", 
                        method=lambda x:mean(x[-10:])),
                    self.timer.query_time_by_name("train forward", 
                        method=lambda x:mean(x[-10:])),
                    self.timer.query_time_by_name("train backward", 
                        method=lambda x:mean(x[-10:])),
                ))
            
            # eval loop
            with torch.no_grad():
                self.model.set_mode(False)
                acc_valid = 0
                for x, quer_y in valid_task:
                    x = x.to(DEVICE)
                    x = self.model(x)
                    supp_x, quer_x = x[:supp_end], x[supp_end:]

                    supp_x = supp_x.view(5, 5, supp_x.size(-1))
                    logit = decode_label(supp_x, quer_x).cpu().numpy()
                    acc_valid += (logit.argmax(1) == np.array(quer_y)).mean()
                acc_valid /= len(valid_task)
                LOGGER.info("epoch %2d valid mean acc %.6f" % (total_epoch, 
                    acc_valid))
            
            if best_valid < acc_valid:
                # save the best model
                best_param = pickle.dumps(self.model.state_dict())
                best_valid = acc_valid

        self.model.load_state_dict(pickle.loads(best_param))
        return MyLearner(self.model.cpu())


class MyLearner(Learner):

    def __init__(self, model: Wrapper = None) -> None:
        """ Defines the learner initialization.
        
        Args:
            model (Wrapper, optional): Learner meta-trained by the MetaLearner. 
                Defaults to None.
        """
        super().__init__()
        self.model = model

    @torch.no_grad()
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
        self.model.to(DEVICE)
        X_train, y_train, _, n, k = support_set
        X_train, y_train = X_train, y_train
        
        return MyPredictor(self.model, X_train, y_train, n, k)

    def save(self, path_to_save: str) -> None:
        """ Saves the learning object associated to the Learner. 
        
        Args:
            path_to_save (str): Path where the learning object will be saved.
        """
        torch.save(self.model, os.path.join(path_to_save, "model.pt"))
 
    def load(self, path_to_load: str) -> None:
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in self.save().
        
        Args:
            path_to_load (str): Path where the Learner is saved.
        """
        if self.model is None:
            self.model = torch.load(os.path.join(path_to_load, 'model.pt'))
    
    
class MyPredictor(Predictor):

    def __init__(self, 
                 model: Wrapper, 
                 supp_x: torch.Tensor, 
                 supp_y: torch.Tensor, 
                 n: int, 
                 k: int) -> None:
        """Defines the Predictor initialization.

        Args:
            model (Wrapper): Learner meta-trained by the MetaLearner.
            supp_x (torch.Tensor): Tensor of labeled images.
            supp_y (torch.Tensor): Tensor of encoded labels.
            n (int): Number of classes.
            k (int): Number of examples per class.
        """
        super().__init__()
        self.model = model
        self.other = [supp_x, supp_y, n, k]

    @torch.no_grad()
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
        query_set = query_set
        supp_x, supp_y, n, k = self.other
        supp_x = supp_x[supp_y.sort()[1]]
        end = supp_x.size(0)
        # to avoid too much gpu memory cost
        x = torch.cat([supp_x, query_set])
        begin_idx = 0
        xs = []
        while begin_idx < x.size(0):
            xs.append(self.model(x[begin_idx: begin_idx + 64].to(
                DEVICE)).cpu())
            begin_idx += 64
        x = torch.cat(xs)
        supp_x, quer_x = x[:end], x[end:]
        supp_x = supp_x.view(n, k, supp_x.size(-1))
        return decode_label(supp_x, quer_x).cpu().numpy()
