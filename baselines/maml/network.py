import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.initializers import GlorotUniform

tf.random.set_seed(1234)


def conv_net(nbr_classes: int = 5, img_size: int = 28):
    """ Reproduce the CNN used in the MAML paper. It was originally designed in
    Vinyals and al. (2016).

    Args:
        nbr_classes (int): Number of classes per training episode. Defaults 
            to 5.
        img_size (int): Size of the input images. Defaults to 28. 

    Returns:
        model: Generated CNN.
    """
    
    model = Sequential()
     
    model.add(Conv2D(64, (3, 3), strides = (1, 1), activation = "relu", 
        input_shape = (img_size, img_size, 3), 
        kernel_initializer = GlorotUniform(seed = 1234)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    for _ in range(2):
        model.add(Conv2D(64, (3, 3), strides = (1, 1), activation = "relu", 
            kernel_initializer = GlorotUniform(seed = 1234)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    
    model.add(Dense(nbr_classes, activation = "softmax", 
        kernel_initializer = GlorotUniform(seed = 1234))) 
    return model
