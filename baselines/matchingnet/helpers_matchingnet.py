import torch
import torch.nn as nn
from typing import List, Tuple


def process_support_set(model: nn.Module, 
                        weights: List[torch.Tensor],
                        X_train: torch.Tensor, 
                        y_train: torch.Tensor, 
                        num_classes: int,
                        dev: torch.device) -> Tuple[torch.Tensor,torch.Tensor]:
    """ Process the support set following the Matching Networks strategy.

    Args:
        model (nn.Module): Model to be used.
        weights (List[torch.Tensor]): Weights to be used by the model.
        X_train (torch.Tensor): Support set images.
        y_train (torch.Tensor): Support set labels.
        num_classes (int): Number of classes to predict.
        dev (torch.device): Device where the data is located.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Normalized embeddings and encoded 
            matrix.
    """
    # Compute input embeddings
    support_embeddings = model.forward_weights(X_train, 
        weights, embedding=True)
    
    # Normalize embeddings
    s_norm = support_embeddings/support_embeddings.norm(
        dim=1).unsqueeze(1)

    # Make one-hot encoded matrix of 
    y = torch.zeros((len(X_train), num_classes), device=dev)
    y[torch.arange(len(X_train)), y_train] = 1
    
    return s_norm, y

def process_query_set(model: nn.Module, 
                      weights: List[torch.Tensor], 
                      X_test: torch.Tensor, 
                      s_norm: torch.Tensor, 
                      y: torch.Tensor) -> torch.Tensor:
    """ Process the query set following the Matching Networks strategy.

    Args:
        model (nn.Module): Model to be used.
        weights (List[torch.Tensor]): Weights to be used by the model.
        X_test (torch.Tensor): Query set images.
        s_norm (torch.Tensor): Normalized support embeddings.
        y (torch.Tensor): Encoded support matrix.

    Returns:
        torch.Tensor: Cosine similarities.
    """
    # Compute input embeddings
    query_embeddings = model.forward_weights(X_test, 
        weights, embedding=True)
    
    # Normalize embeddings
    q_norm = query_embeddings/query_embeddings.norm(
        dim=1).unsqueeze(1)

    # Matrix of cosine similarity scores (i,j)-entry is similarity of 
    # query input i to support example j
    cosine_similarities = torch.mm(s_norm, q_norm.transpose(0,1)).t()

    # Cosine similarities
    out = torch.mm(cosine_similarities, y)
    
    return out
