""" Defines the API used in the cross-domain meta-learning challenge. Please 
check the dedicated notebook tutorial for details.
"""

class MetaLearner():
    """ Define the meta-learning algorithm we want to use, through its methods.
    It is an abstract class so one has to overide the core methods depending 
    on the algorithm.
    """
    
    def __init__(self, N_ways: int, total_train_classes: int) -> None:
        """ Defines the meta-learning algorithm's parameters. For example, one 
        has to define what would be the learner meta-learner's architecture. 
        For instance, one could use the Keras API to define models.
        
        Args:
            N_ways (int): Number of ways used for training. It is only used 
                when data format is 'episode'.
            total_train_classes (int): Total number of classes that can be seen 
                during meta-training.
        """
        self.N_ways = N_ways
        self.total_train_classes = total_train_classes

    def meta_fit(self, meta_train_generator, meta_valid_generator):
        """ Uses the meta-dataset to fit the meta-learner's parameters. A 
        meta-dataset can be an episode or a batch images.
        
        Args:
            meta_train_generator: Function that generates the training data.
                The generated can be an episode (N-ways any-shot learning task) 
                or a batch of images with labels.
            meta_valid_generator: Function that generates the validation data.
                The generated data always come in form of any-ways any-shot 
                learning tasks.
                
        Returns:
            Learner: Resulting learner ready to be trained and evaluated on new
                unseen tasks.
                
        Note: 
        Each episode is an object with the following attributes: 
            num_ways (int): Number of ways (classes) for the support set. 
            num_shots (int): Number of shots (images per class) for the support 
                set.
            support_set (tuple): Support set for the episode. The format of the 
                set is (np.ndarray, np.ndarray), where the first array 
                corresponds to the images and has a shape of 
                (num_ways*num_shots x 128 x 128 x 3) while the second array 
                corresponds to the labels and its shape is 
                (num_ways*num_shots, ). 
            query_set (tuple): Query set for the episode. The format of the set
                is (np.ndarray, np.ndarray), where the first array 
                corresponds to the images and has a shape of 
                (query_size x 128 x 128 x 3) while the second array corresponds
                to the labels and its shape is (query_size, ). 
                 
        On the other hand each batch is composed of images and labels in the 
        following format: (np.ndarray, np.ndarray). The first array corresponds
        to the images and has a shape of (batch_size x 128 x 128 x 3) while the
        second array corresponds to the labels and its shape is (batch_size, ).
        
        
        """
        raise NotImplementedError(("You should implement the meta_fit method "
            + f"for the MetaLearner class."))
    

class Learner():
    """ This class represents the learner returned at the end of the 
    meta-learning procedure.
    """
    
    def __init__(self):
        pass

    def fit(self, dataset_train):
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
        raise NotImplementedError(("You should implement the fit method for "
            + "the Learner class."))
    
    def save(self, path_to_save: str) -> None:
        """ Saves the learning object associated to the Learner. It could be 
        a neural network for example. 
        
        Args:
            path_to_save (str): Path where the Learner will be saved

        Note: It is mandatory to allow saving the Learner as a file in 
        path_to_save. Otherwise, it won't be a valid submission.
        """
        raise NotImplementedError(("You should implement the save method for "
            + "the Learner class."))

    def load(self, path_to_model: str) -> None:
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in save().
        
        Args:
            path_to_model (str): Path where the Learner is saved
        """
        raise NotImplementedError(("You should implement the load method for "
            + "the Learner class."))


class Predictor():
    """ This class represents the predictor returned at the end of the 
    Learner's fit method. 
    """
    
    def __init__(self):
        pass

    def predict(self, dataset_test):
        """ Given a dataset_test, predicts the probabilities associated to the 
        provided images.
        
        Args:
            dataset_test: Array of unlabelled image examples of shape 
                (query_size x 128 x 128 x 3).
        
        Returns:
            np.ndarray: Predicted probs for all images. The array must be of 
                shape (query_size, N_ways).
        """
        raise NotImplementedError(("You should implement the predict method "
            + "for the Predictor class."))

