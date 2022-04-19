"""Transfer baseline.
Here, we consider the transfer learning approach. We first load a model 
pre-trained on ImageNet. We freeze the layers associated to the projected 
images and we fine-tune a classifer on top of this embedding function. 
"""
import os
import logging
import datetime
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras

from api import MetaLearner, Learner, Predictor

class MyMetaLearner(MetaLearner):
    
    def __init__(self,
                 N_ways: int, 
                 total_train_classes: int,
                 iterations: int = 250,
                 batch_size: int = 100,
                 img_size: int = 71) -> None:
        """
        Args:
            N_ways (int): Number of ways used for training. It is only used 
                when data format is 'episode'.
            total_train_classes (int): Total number of classes that can be seen 
                during meta-training. 
            iterations (int): Number of iterations consider at training time. 
                Defaults to 10.
            batch_size (int): Number of examples per batch. Defaults to 100.
            img_size (int): Size of the images that will be processed, the 
                shape is (img_size, img_size, 3). Defaults to 71.
        """
        super().__init__(N_ways, total_train_classes)
        self.iterations = iterations
        self.batch_size = batch_size
        self.img_size = img_size

        # Initialize pre-trained model
        self.base_model = keras.applications.Xception(
            weights = "imagenet",
            input_shape = (self.img_size, self.img_size, 3),
            include_top = False
        )
        self.base_model.trainable = False
        inputs = keras.Input(shape = (self.img_size, self.img_size, 3))
        x = self.base_model(inputs, training=True)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self.total_train_classes)(x)
        self.model = keras.Model(inputs, outputs)

        self.optimizer = keras.optimizers.Adam()

        self.loss = keras.losses.SparseCategoricalCrossentropy()

        # Summary Writers
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_summary_writer = tf.summary.create_file_writer(
            f"logs/transfer/gradient_tape/{current_time}/meta-train")
        self.valid_summary_writer = tf.summary.create_file_writer(
            f"logs/transfer/gradient_tape/{current_time}/meta-valid")

        # Statistics tracker
        self.train_loss = tf.keras.metrics.Mean(name = "train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name = "train_accuracy")
        
        self.valid_loss = tf.keras.metrics.Mean(name = "valid_loss")
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name = "valid_accuracy")

    def meta_fit(self, meta_train_generator, meta_valid_generator) -> Learner:
        """ Train the classfier created on top of the pre-trained embedding
        layers.

        Args:
            meta_train_generator: Function that generates the training data.
                The generated can be an episode (N-ways any-shot learning task) 
                or a batch of images with labels.
            meta_valid_generator: Function that generates the validation data.
                The generated data always come in form of any-ways any-shot 
                learning tasks.
        
        Returns:
            Learner: Resulting learner that stores the current embedding 
                function (Neural Network) of this MetaLearner. 
        """
        logging.info("Starting meta-fit for the transfer baseline ...")
        sample_data = next(meta_train_generator(1, self.batch_size))
        logging.info(f"Images shape: {sample_data[0].shape}")
        logging.info(f"Labels shape: {sample_data[1].shape}")
        
        for i, (images, labels) in enumerate(meta_train_generator(
                self.iterations, self.batch_size)):
            
            # Training
            with tf.GradientTape() as tape:
                preds = self.model(self.resize_images(images))
                loss = self.loss(labels, preds)
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, 
                self.model.trainable_weights))
            
            self.train_accuracy.update_state(labels, preds)
            self.train_loss.update_state(loss)
            
            logging.info(f"Iteration #{i+1} - Loss : {loss.numpy()}")
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar("Train Loss", self.train_loss.result(),
                    step = i+1)
                tf.summary.scalar("Train Accuracy", 
                    self.train_accuracy.result(), step = i+1)
            
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
                
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

        return MyLearner(self.model, self.img_size)

    def resize_images(self, img_set: np.ndarray) -> np.ndarray:
        """ Resize the images to the expected image size.
        
        Args:
            img_set (np.ndarray): Image set of shape (set_size x 128 x 128 x 3)
        
        Returns:
            np.ndarray: Resized and rotated image set.
        """
        new_img_set = []
        for image in img_set:
            new_img = cv2.resize(image, (self.img_size, self.img_size), 
                interpolation=cv2.INTER_AREA)
            new_img_set.append(new_img)      
        return np.asarray(new_img_set)
    
    def evaluate(self, meta_valid_generator) -> None:
        """ Evaluates the current meta-learner with episodes generated from the
        meta-validation split. The number of episodes used to compute the 
        an average accuracy is set to 20.

        Args:
            meta_valid_generator: Generator of validation tasks.
        """
        for task in meta_valid_generator(20): 
            learner = MyLearner(self.model, self.img_size)
            dataset_train = (task.support_set[0], task.support_set[1], 
                task.num_ways, task.num_shots)
            
            predictor = learner.fit(dataset_train)
            preds = predictor.predict(task.query_set[0])
            loss = self.loss(task.query_set[1], preds)
            
            self.valid_loss.update_state(loss)
            self.valid_accuracy.update_state(task.query_set[1], preds)
        
        logging.info(f"Meta-Valid accuracy:{self.valid_accuracy.result():.3f}")


class MyLearner(Learner):
    def __init__(self, 
                 model = None,
                 img_size: int = 71) -> None:
        """ If no base model is provided, we create a neural network with 
        randomly initialized weights.
        
        Args:
            model: Meta-trained neural network. Defaults to None.
            img_size (int): Size of the images that will be processed, the 
                shape is (img_size, img_size, 3). Defaults to 71.
        """
        super().__init__()
        self.img_size = img_size
        
        if model == None:
            self.base_model = keras.applications.Xception(
                weights = "imagenet",
                input_shape = (img_size, img_size, 3),
                include_top = False
            )
            self.base_model.trainable = False
            inputs = keras.Input(shape = (img_size, img_size, 3))
            x = self.base_model(inputs, training=False)
            outputs = keras.layers.GlobalAveragePooling2D()(x)
            self.learner = keras.Model(inputs, outputs)
            
        else: 
            self.learner = keras.models.clone_model(model)
            self.learner.set_weights(model.get_weights())
            self.learner = keras.Model(inputs=model.input, 
                outputs=model.layers[-2].output)
            
        self.x = self.learner.output
            
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.SparseCategoricalCrossentropy()

    def fit(self, dataset_train) -> Predictor:
        """ Fine-tunes the current model with the support examples of a new 
        unseen task. 

        Args:
            dataset_train: Support set of a task. The data arrive in the 
                following format (X_train, y_train, n_ways, k_shots). X_train 
                is the array of labeled imaged of shape 
                (n_ways*k_shots x 128 x 128 x 3), y_train are the encoded
                labels (int) for each image in X_train, n_ways (int) are the 
                number of classes and k_shots (int) the number of examples per 
                class.
                
        Returns:
            Predictor: The resulted Predictor object that is initialized with 
                the fine-tuned Learner's neural network weights.
        """
        images, labels, num_ways, _ =  dataset_train 
        resized_images = []
        for image in images:
            resized_images.append(cv2.resize(image, (self.img_size, 
                self.img_size), interpolation=cv2.INTER_AREA))
        resized_images = np.asarray(resized_images)
        
        outputs = keras.layers.Dense(num_ways, activation = "softmax")(self.x)
        model = keras.Model(inputs = self.learner.input, outputs = outputs)
        
        logging.debug("Fitting a task ...")
        for _ in range(5):
            logging.debug(f"Image shape: {resized_images.shape}")
            logging.debug(f"Labels shape: {labels.shape}")
            with tf.GradientTape() as tape :
                preds = model(resized_images)
                loss = self.loss(labels, preds)
            logging.debug(f"[FIT] Loss on support set : {loss.numpy()}")
            grads = tape.gradient(loss, model.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, model.trainable_weights))

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
        self.x = self.learner.output

    
class MyPredictor(Predictor):
    
    def __init__(self, 
                 model,
                 img_size: int) -> None:
        """
        Args: 
            model: Fine-tuned neural network.
            img_size (int): Size of the images that will be processed (should 
                be the same as the learner).
        """
        super().__init__()
        self.model = model
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
        preds = self.model(np.asarray(resized_images))
        return preds.numpy()

