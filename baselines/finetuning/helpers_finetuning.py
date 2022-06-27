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
