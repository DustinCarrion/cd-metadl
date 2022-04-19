""" This script contains the implementation of the MAML algorithm designed 
by Chelsea Finn et al. (https://arxiv.org/pdf/1703.03400).
"""
import datetime
import os 
import numpy as np
import cv2
import imutils
import tensorflow as tf
from network import conv_net

from api import MetaLearner, Learner, Predictor

np.random.seed(1234)
tf.random.set_seed(1234)


class MyMetaLearner(MetaLearner):

    def __init__(self,
                 N_ways: int, 
                 total_train_classes: int,
                 meta_iterations: int = 500,
                 meta_batch_size: int = 20,
                 inner_steps: int = 5,
                 img_size: int = 28) -> None:
        """
        Args:
            N_ways (int): Number of ways used for training. It is only used 
                when data format is 'episode'.
            total_train_classes (int): Total number of classes that can be seen 
                during meta-training. 
            meta_iterations (int): Number of iterations to consider at 
                meta-train time. Defaults to 100.
            meta_batch_size (int): Number of episodes evaluated in one 
                iteration. Defaults to 10.
            inner_steps (int): Gradient steps for the inner loop. Defaults to 5
            img_size (int): Size of the images that will be processed, the 
                shape is (img_size, img_size, 3). Defaults to 28.
        """
        super().__init__(N_ways, total_train_classes)
        self.meta_iterations = meta_iterations
        self.meta_batch_size = meta_batch_size
        self.inner_steps = inner_steps
        self.img_size = img_size

        self.meta_learner = conv_net(self.N_ways, self.img_size)
        self.current_meta_weights = self.meta_learner.get_weights()
        
        self.meta_learner_optimizer = tf.keras.optimizers.SGD(
            learning_rate = 0.01)
        self.task_optimizer = tf.keras.optimizers.SGD(learning_rate = 0.05)
        
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

        # Summary Writers 
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_summary_writer = tf.summary.create_file_writer(
            f"logs/MAML/gradient_tape/{current_time}/meta-train")
        self.valid_summary_writer = tf.summary.create_file_writer(
            f"logs/MAML/gradient_tape/{current_time}/meta-valid") 
        
        # Statistics tracker
        self.train_loss = tf.keras.metrics.Mean(name = "train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name = "train_accuracy")
        
        self.valid_loss = tf.keras.metrics.Mean(name = "valid_loss") 
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="valid_accuracy")

    def meta_fit(self, meta_train_generator, meta_valid_generator) -> Learner:
        """ Generates episodes from the meta-train split and updates the 
        meta-learner's weights according to the learning algorithm
        described in the original paper. Every 10 iterations, we evaluate the 
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
            Learner: A Learner initialized with the meta-learner's weights.
        """
        for i in range(self.meta_iterations):
            self.current_meta_weights = self.meta_learner.get_weights()
            
            task_weights = []
            meta_batch_query_imgs = []
            meta_batch_query_lbls = []
            
            # Inner loop
            for task in meta_train_generator(self.meta_batch_size):
                task_weights.append(self.optimize_objective(
                    self.apply_modifications(task.support_set[0]), 
                    task.support_set[1]))
                meta_batch_query_imgs.append(self.apply_modifications(
                    task.query_set[0]))
                meta_batch_query_lbls.append(task.query_set[1])
                
            # Outer loop
            self.meta_optimize(task_weights, meta_batch_query_imgs, 
                meta_batch_query_lbls)

            with self.train_summary_writer.as_default():
                tf.summary.scalar("Train Loss", self.train_loss.result(), 
                    step = i+1)
                tf.summary.scalar("Train Accuracy", 
                    self.train_accuracy.result(), step = i+1)
            
            self.train_accuracy.reset_states()
            self.train_loss.reset_states()
            
            # Validation
            if (i+1) % 10 == 0:
                self.evaluate(meta_valid_generator)
                
                with self.valid_summary_writer.as_default():
                    tf.summary.scalar("Validation Loss", 
                        self.valid_loss.result(), step = i+1)
                    tf.summary.scalar("Validation Accuracy", 
                        self.valid_accuracy.result(), step = i+1)
                
                self.valid_loss.reset_states()
                self.valid_accuracy.reset_states()
            
        return MyLearner(self.meta_learner, self.img_size)

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

    def optimize_objective(self, 
                           support_imgs: np.ndarray, 
                           support_lbls: np.ndarray) -> list:
        """ Optimize the meta-leaner's weights for the specified task.

        Args:
            support_imgs (np.ndarray): Images of the support set of the current 
                episode.
            support_lbls (np.ndarray): Labels of the support set of the current 
                episode.

        Returns:
            list: List of optimized weights for the current episode.
        """
        # Reset the meta-leaner's weights
        self.meta_learner.set_weights(self.current_meta_weights)
        
        # Optimize the meta-leaner's weights for the current episode
        for _ in range(self.inner_steps):
            with tf.GradientTape() as tape:
                preds = self.meta_learner(support_imgs, training=True)
                loss = self.loss(support_lbls, preds)
            
            grads = tape.gradient(loss, self.meta_learner.trainable_variables)
            self.task_optimizer.apply_gradients(zip(grads, 
                self.meta_learner.trainable_variables))
        
        return self.meta_learner.get_weights()

    def meta_optimize(self, 
                      task_weights: list, 
                      meta_batch_query_imgs: list, 
                      meta_batch_query_lbls: list) -> None:
        """ Optimize the meta-leaner's weights considering all the evaluated 
        episodes.

        Args:
            task_weights (list): _description_
            meta_batch_query_imgs (list): Query set images of the meta-batch.
            meta_batch_query_lbls (list): Query set labels of the meta-batch.
        """
        losses = []
        with tf.GradientTape() as tape:
            for i, (query_imgs, query_lbls) in enumerate(zip(
                meta_batch_query_imgs, meta_batch_query_lbls)):
                # Assign the task weights
                self.meta_learner.set_weights(task_weights[i])
                
                preds = self.meta_learner(query_imgs, training=True)
                loss = self.loss(query_lbls, preds)
                losses.append(loss)
                self.train_accuracy.update_state(query_lbls, preds)
            mean_loss = tf.reduce_mean(losses)
                
        self.train_loss.update_state(mean_loss)
        
        # Assign the current meta learner weights before updating them
        self.meta_learner.set_weights(self.current_meta_weights)
        grads = tape.gradient(mean_loss, self.meta_learner.trainable_variables)
        self.meta_learner_optimizer.apply_gradients(zip(grads, 
                self.meta_learner.trainable_variables))

    def evaluate(self, meta_valid_generator) -> None:
        """ Evaluates the current meta-learner with episodes generated from the
        meta-validation split. The number of episodes used to compute the an 
        average accuracy is set to 20.
        
        Args:
            meta_valid_generator: Generator of validation tasks.
        """
        for task in meta_valid_generator(20):
            learner = MyLearner(self.meta_learner, self.img_size)
            dataset_train = (task.support_set[0], task.support_set[1], 
                task.num_ways, task.num_shots)
            
            predictor = learner.fit(dataset_train)
            preds = predictor.predict(task.query_set[0])
            loss = self.loss(task.query_set[1], preds)
            
            self.valid_loss.update_state(loss)
            self.valid_accuracy.update_state(task.query_set[1], preds)

    
class MyLearner(Learner):
    
    def __init__(self, 
                 learner = None,
                 img_size: int = 28) -> None:
        """ If no learner is provided, we create a neural network with randomly 
        initialized weights.
        
        Args:
            learner: Meta-trained learner. Defaults to None.
            img_size (int): Size of the images that will be processed (should 
                be the same as the meta-learner). Defaults to 28. 
        """
        super().__init__()
        self.img_size = img_size
        if learner == None:
            self.learner = conv_net(10, img_size=img_size)
        else: 
            self.learner = tf.keras.models.clone_model(learner)
            self.learner.set_weights(learner.get_weights())

        # Learning procedure parameters
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.05)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
        self.max_iterations = 20

    def fit(self, dataset_train) -> Predictor:
        """ Adjust the weights of the learner to the current episode. We need 
        to know the number of classes (N_ways) to update the output layer of 
        the learner.

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
        images, labels, num_ways, _ =  dataset_train 
        resized_images = []
        for image in images:
            resized_images.append(cv2.resize(image, (self.img_size, 
                self.img_size), interpolation=cv2.INTER_AREA))
        
        # Adapt model to current task
        custom_layer = tf.keras.layers.Dense(num_ways, activation = "softmax", 
            kernel_initializer = tf.keras.initializers.GlorotUniform(
            seed = 1234))(self.learner.layers[-2].output)
        model = tf.keras.Model(inputs=self.learner.input, outputs=custom_layer)
        
        # Train model
        for i in range(self.max_iterations): 
            with tf.GradientTape() as tape:
                preds = model(np.asarray(resized_images), training=True)
                loss = self.loss(labels, preds)
                self.accuracy.update_state(labels, preds)
                
            grads = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, 
                model.trainable_variables))
        
            if self.accuracy.result().numpy() > 0.8 and i > 2:
                break
        
        return MyPredictor(model, self.img_size)
    
    def save(self, path_to_save: str) -> None:
        """ Saves the learner as a tensorflow checkpoint.
        
        Args:
            path_to_save (str): Path where the Learner will be saved
        """
        ckpt_file = os.path.join(path_to_save, "learner.ckpt")
        self.learner.save_weights(ckpt_file) 
        
    def load(self, path_to_model: str) -> None:
        """ Loads the learner from a tensorflow checkpoint.
        
        Args:
            path_to_model (str): Path where the Learner is saved
        """
        ckpt_path = os.path.join(path_to_model, "learner.ckpt")
        self.learner.load_weights(ckpt_path)
        
        
class MyPredictor(Predictor):
    
    def __init__(self,
                 learner,
                 img_size: int):
        """
        Args:
            learner: Optimized learner.
            img_size (int): Size of the images that will be processed (should 
                be the same as the learner).
        """
        super().__init__()
        self.learner = learner
        self.img_size = img_size
    
    def predict(self, dataset_test: np.ndarray) -> np.ndarray:
        """ Given a dataset_test, predicts the labels probabilities associated 
        to the provided images. 

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
        preds = self.learner(np.asarray(resized_images))
        return preds.numpy()

