import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


def optimize(model: nn.Module, 
             optimizer: torch.optim, 
             X: torch.Tensor, 
             y: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """ Apply an optimization step using the specified model, optimizer and 
    data.

    Args:
        model (nn.Module): Model to be optimized.
        optimizer (torch.optim): Optimizer to be used.
        X (torch.Tensor): Input data.
        y (torch.Tensor): Ground truth labels.
        
    Returns:
        Tuple[torch.Tensor, float]: Network output and loss.
    """
    optimizer.zero_grad()
    out = model(X)
    loss = model.criterion(out, y)
    loss.backward()
    optimizer.step()
    return out, loss.item()


def get_batch(X: torch.Tensor, 
              y: torch.Tensor, 
              batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Get a batch of the specified size from the specified data.

    Args:
        X (torch.Tensor): Images.
        y (torch.Tensor): Labels.
        batch_size (int): Desired batch size.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batch of the specified size.
    """
    batch_indices = np.random.randint(0, X.size()[0], batch_size)
    X_batch, y_batch = X[batch_indices], y[batch_indices]
    return X_batch, y_batch


def process_support_set(model: nn.Module, 
                        X_train: torch.Tensor, 
                        y_train: torch.Tensor, 
                        num_classes: int) -> torch.Tensor:
    """ Process the support set following the NCC strategy.

    Args:
        model (nn.Module): Model to be used.
        X_train (torch.Tensor): Support set images.
        y_train (torch.Tensor): Support set labels.
        num_classes (int): Number of classes to predict.

    Returns:
        torch.Tensor: Support prototypes.
    """
    # Compute input embeddings
    support_embeddings = model(X_train, embedding=True)
    
    # Compute prototypes
    prototypes = torch.zeros((num_classes, support_embeddings.size(1)), 
        device=X_train.device)
    for i in range(num_classes):
        mask = y_train == i
        prototypes[i] = (support_embeddings[mask].sum(dim=0) / 
            torch.sum(mask).item())
        
    return prototypes


def process_query_set(model: nn.Module, 
                      X_test: torch.Tensor, 
                      prototypes: torch.Tensor) -> torch.Tensor:
    """ Process the query set following the Matching Networks strategy.

    Args:
        model (nn.Module): Model to be used.
        X_test (torch.Tensor): Query set images.
        prototypes (torch.Tensor): Support prototypes.

    Returns:
        torch.Tensor: Distances to prototypes.
    """
    # Compute input embeddings
    query_embeddings = model(X_test, embedding=True)

    # Create distance matrix (negative predictions)
    distance_matrix = (torch.cdist(query_embeddings.unsqueeze(0), 
        prototypes.unsqueeze(0))**2).squeeze(0) 
    out = -1 * distance_matrix
    
    return out
