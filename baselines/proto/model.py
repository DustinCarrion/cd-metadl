""" Prototypical networks implementation. This implementation is based on 
the original Prototypical networks paper J. Snell et al. 2017 
(https://arxiv.org/pdf/1703.05175).
"""
import os
import logging
import datetime
import numpy as np
import cv2
import imutils
import tensorflow as tf
from utils import create_proto_shells
from network import conv_net

from api import MetaLearner, Learner, Predictor

np.random.seed(1234)
tf.random.set_seed(1234)


class MyMetaLearner(MetaLearner):

    def __init__(self, 
                 N_ways: int, 
                 total_train_classes: int,
                 meta_iterations: int = 2500,
                 distance_fn = tf.norm,
                 embedding_dim: int = 64,
                 img_size: int = 28) -> None:
        """
        Args:
            N_ways (int): Number of ways used for training. It is only used 
                when data format is 'episode'.
            total_train_classes (int): Total number of classes that can be seen 
                during meta-training. 
            meta_iterations (int): Number of episodes to consider at meta-train
                time. Defaults to 500.
            distance_fn: Distance function used for the proto-networks. 
                Defaults to tf.norm.
            embedding_dim (int): Embedding dimension. Defaults to 64.
            img_size (int): Size of the images that will be processed, the 
                shape is (img_size, img_size, 3). Defaults to 28.
        """
        super().__init__(N_ways, total_train_classes)
        self.meta_iterations = meta_iterations
        self.distance_fn = distance_fn
        self.embedding_dim = embedding_dim
        self.img_size = img_size

        self.embedding_fn = conv_net(self.img_size)

        self.learning_rate = 1e-3
        self.optimizer = tf.optimizers.Adam(learning_rate = self.learning_rate)
        self.loss = 0

        # Summary Writers
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_summary_writer = tf.summary.create_file_writer(
            f"logs/proto/gradient_tape/{current_time}/meta-train")
        self.valid_summary_writer = tf.summary.create_file_writer(
            f"logs/proto/gradient_tape/{current_time}/meta-valid")
        
        # Statistics tracker
        self.train_loss = tf.keras.metrics.Mean(name = "train_loss")
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="valid_accuracy")

    def meta_fit(self, meta_train_generator, meta_valid_generator) -> Learner:
        """ Generates episodes from the meta-train split and updates the 
        embedding function (Neural network) according to the learning algorithm
        described in the original paper. Every 50 tasks, we evaluate the 
        current meta-learner with episodes generated from the meta-validation 
        split.
        
        Args:
            meta_train_generator: Function that generates the training data.
                The generated can be an episode (N-ways any-shot learning task) 
                or a batch of images with labels.
            meta_valid_generator: Function that generates the validation data.
                The generated data always come in form of any-ways any-shot 
                learning tasks.
        
        Returns:
            Learner: A Learner that stores the current embedding function 
                (Neural Network) of this MetaLearner.
        """
        logging.info('Starting meta-fit for the proto-net ...')
        for i, task in enumerate(meta_train_generator(self.meta_iterations)):
            self.prototypes = create_proto_shells(task.num_ways, 
                self.embedding_dim)
            
            self.compute_prototypes(self.apply_modifications(
                task.support_set[0]), task.support_set[1], task.num_ways, 
                task.num_shots)
            
            self.meta_optimize(self.apply_modifications(task.query_set[0]), 
                task.query_set[1])
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar("Train Loss", self.train_loss.result(),
                    step = i+1)

            self.train_loss.reset_states()
            
            # Validation
            if (i+1) % 10 == 0:
                self.evaluate(meta_valid_generator)
                
                with self.valid_summary_writer.as_default():
                    tf.summary.scalar("Validation Accuracy", 
                        self.valid_accuracy.result(), step = i+1)
                self.valid_accuracy.reset_states()
            
            if (i+1) % 2000 == 0:
                self.learning_rate /= 2
                self.optimizer = tf.keras.optimizers.Adam(learning_rate = 
                    self.learning_rate)
                logging.info(f"New learning rate: {self.learning_rate}")
            
        return MyLearner(self.embedding_fn, self.embedding_dim, self.img_size)

    def apply_modifications(self, img_set: np.ndarray) -> np.ndarray:
        """ Resize the images to the expected image size and randomly rotate 
        them. The rotation angle can be: 0, 90, 180, or 270. The rotation is 
        applied to reproduce the data augmentation performed in the original 
        Prototypical Networks.
        
        Args:
            img_set (np.ndarray): Image set of shape (set_size x 128 x 128 x 3)
        
        Returns:
            np.ndarray: Resized and rotated image set.
        """
        new_img_set = []
        for image in img_set:
            curr_img = cv2.resize(image, (self.img_size, self.img_size), 
                interpolation=cv2.INTER_AREA)
            angle = np.random.choice([0, 90, 180, 270])
            curr_img = imutils.rotate(curr_img, angle)
            new_img_set.append(curr_img)      
        return np.asarray(new_img_set)

    def compute_prototypes(self, 
                           support_imgs: np.ndarray, 
                           support_lbls: np.ndarray, 
                           N_ways: int,
                           K_shots: int) -> None:
        """ Computes the prototypes of the support set examples. They are 
        computed as the average of the embedding projections of the examples 
        within each class.

        Args:
            support_imgs (np.ndarray): Images of the support set.
            support_lbls (np.ndarray): Labels of the support set.
            N_ways (int): Number of ways (classes). 
            K_shots (int): Number of shots (images per class).
        """
        logging.debug("Computing prototypes ...")
        logging.debug(f"A prototype shape : {self.prototypes[0].shape}")
        logging.debug(f"Images shape : {support_imgs.shape}")
        logging.debug(f"Labels shape : {support_lbls.shape}")
        proj_imgs = self.embedding_fn(support_imgs, training = True)
        for i, label in enumerate(support_lbls): 
            logging.debug(f"Label : {label}")
            self.prototypes[label] += proj_imgs[i]

        for i in range(N_ways):
            self.prototypes[i] /= K_shots
        
        logging.debug(f"Prototypes after computing them : {self.prototypes}")

    def meta_optimize(self, 
                      query_imgs: np.ndarray, 
                      query_lbls: np.ndarray) -> None:
        """ Computes the distance between prototypes and query examples and 
        update the loss according to each of these values. The loss we used is 
        the one derived in the original paper https://arxiv.org/pdf/1703.05175. 
        (Page 2, equation #2)

        Args: 
            query_imgs (np.ndarray): Images of the query set.
            query_lbls (np.ndarray): Labels of the query set.
        """
        cste = 1/len(query_imgs)
        
        logging.debug(f"Images shape : {query_imgs.shape}")
        logging.debug(f"Labels shape : {query_lbls.shape}")
        with tf.GradientTape() as tape:
            proj_imgs = self.embedding_fn(query_imgs, training = True)
            logging.debug(f"Projected images shape : {proj_imgs.shape}")
            
            for i, label in enumerate(query_lbls):
                proj_image = proj_imgs[i]

                # Distance of proj image to corresponding prototype 
                tmp1 = self.distance_fn(proj_image - self.prototypes[label])
                # Log sum exp of distancees between projection and prototypes
                tmp2 = tf.math.reduce_logsumexp(-self.distance_fn(tf.squeeze(
                    proj_image - self.prototypes, axis=1), axis=1))
                
                self.loss += cste * (tmp1 + tmp2)
            
            logging.info(f"Loss on a task : {self.loss}")
            self.train_loss.update_state(self.loss)
            
        grads = tape.gradient(self.loss, self.embedding_fn.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, 
            self.embedding_fn.trainable_weights))
        
        self.loss = 0 # Reset loss after a task
    
    def evaluate(self, meta_valid_generator) -> None:
        """ Evaluates the current meta-learner with episodes generated from the
        meta-validation split. The number of episodes used to compute the an 
        average accuracy is set to 20.
        
        Args:
            meta_valid_generator: Generator of validation tasks.
        """
        for task in meta_valid_generator(20):
            learner = MyLearner(self.embedding_fn, self.embedding_dim, 
                self.img_size)
            dataset_train = (task.support_set[0], task.support_set[1], 
                task.num_ways, task.num_shots)
            
            predictor = learner.fit(dataset_train)
            preds = predictor.predict(task.query_set[0])
            self.valid_accuracy.update_state(task.query_set[1], preds)

  
class MyLearner(Learner):

    def __init__(self,
                 embedding_fn = None,
                 embedding_dim: int = 64,
                 img_size: int = 28) -> None:
        """ If no embedding function is provided, we create a neural network
        with randomly initialized weights.
 
        Args:
            embedding_fn: Meta-trained embedding function (Neural Network). 
                Defaults to None.
            embedding_dim (int): Embedding dimension (should be the same as the 
                meta-learner). Defaults to 64.
            img_size (int): Size of the images that will be processed (should 
                be the same as the meta-learner). Defaults to 28.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.img_size = img_size
        if embedding_fn == None:
            self.embedding_fn = conv_net(self.img_size)
        else:
            self.embedding_fn = tf.keras.models.clone_model(embedding_fn)
            self.embedding_fn.set_weights(embedding_fn.get_weights())
            

    def fit(self, dataset_train: tuple) -> Predictor:
        """ Compute the prototypes of the corresponding support set which is 
        dataset_train (support set) in this case. We need to know which the 
        number of classes (N_ways) and the number of shots per class (K_shots) 
        to compute each one of them.

        Args: 
            dataset_train (tuple): Support set of a task. The data arrive in 
                the following format (images, labels, n_ways, k_shots). 
                images is the array of labeled imaged of shape 
                (n_ways*k_shots x 128 x 128 x 3), labels are the encoded
                labels (int) for each image in images, n_ways (int) are the 
                number of classes and k_shots (int) the number of examples per 
                class.
                
        Returns:
            Predictor: The resulted Predictor that has computed prototypes.
        """
        # Preprocess data
        images, labels, num_ways, num_shots =  dataset_train 
        resized_images = []
        for image in images:
            resized_images.append(cv2.resize(image, (self.img_size, 
                self.img_size), interpolation=cv2.INTER_AREA))
        
        # Compute prototypes
        proj_imgs = self.embedding_fn(np.asarray(resized_images)).numpy()
        prototypes = create_proto_shells(num_ways, self.embedding_dim)
        for i, label in enumerate(labels):
            prototypes[label] += proj_imgs[i]

        for i in range(num_ways):
            prototypes[i] /= num_shots

        return MyPredictor(self.embedding_fn, prototypes, num_ways, 
            self.img_size)

    def save(self, path_to_save: str) -> None:
        """ Saves the embedding function as a tensorflow checkpoint.
        
        Args:
            path_to_save (str): Path where the Learner will be saved.
        """
        ckpt_file = os.path.join(path_to_save, "embedding_function.ckpt")
        self.embedding_fn.save_weights(ckpt_file) 

    def load(self, path_to_model: str) -> None:
        """ Loads the embedding function from a tensorflow checkpoint.
        
        Args:
            path_to_model (str): Path where the Learner is saved.
        """
        ckpt_path = os.path.join(path_to_model, "embedding_function.ckpt")
        self.embedding_fn.load_weights(ckpt_path)
    
    
class MyPredictor(Predictor):

    def __init__(self,
                 embedding_fn,
                 prototypes: list,
                 N_ways: int,
                 img_size: int,
                 distance_fn = np.linalg.norm) -> None:
        """
        Args:
            embedding_fn: Trained embedding function (Neural Network). 
            prototypes (list): Prototypes computed using the support set.
            N_ways (int): Number of ways (classes) in the episode. 
            img_size (int): Size of the images that will be processed (should 
                be the same as the learner).
            distance_fn: Distance function to consider for the proto-networks.
        """
        super().__init__()
        self.embedding_fn = embedding_fn
        self.prototypes = prototypes
        self.N_ways = N_ways
        self.img_size = img_size
        self.distance_fn = distance_fn

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:
        """ Given a dataset_test, predicts the labels probabilities associated 
        to the provided images. The prototypes are already computed by the 
        Learner.

        Args:
            dataset_test (np.ndarray): Array of unlabelled image examples of 
                shape (query_size x 128 x 128 x 3).
        
        Returns:
            np.ndarray: Predicted probs for all images. The array must be of 
                shape (query_size, N_ways).
        """
        resized_images = []
        for image in dataset_test:
            resized_images.append(cv2.resize(image, (self.img_size, 
                self.img_size), interpolation=cv2.INTER_AREA))
        proj_imgs = self.embedding_fn(np.asarray(resized_images)).numpy()
        query_size, embedding_dim = proj_imgs.shape

        broadcast_proto = np.broadcast_to(np.expand_dims(np.squeeze(
            self.prototypes), axis=0), [query_size, self.N_ways, embedding_dim
            ])

        broadcast_projections = np.broadcast_to(np.expand_dims(proj_imgs, 
            axis = 1), [query_size, self.N_ways, embedding_dim])

        # Softmax probabilities
        numerator = np.exp(-self.distance_fn(broadcast_projections - 
            broadcast_proto, axis =2))
        denominator = np.repeat(np.sum(numerator, axis=1).reshape(-1,1), 
            self.N_ways, axis=1)
        return numerator/denominator
