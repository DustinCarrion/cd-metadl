import tensorflow as tf

def create_proto_shells(N_ways: int, d: int) -> list:
    """ Create a prototype shell. In an episode, there are N_ways prototypes, 
    i.e. one for each class. For each class, the associated prototype is a 
    d-dimensional vector. 'd' is the embedding dimension.
    
    Args:
        N_ways (int): Number of classes in an episode.
        d (int): Embedding dimension.
    
    Returns:
        list: List with prototype shells.
    """
    proto_shells = [tf.zeros((1, d)) for _ in range(N_ways)]
    return proto_shells
