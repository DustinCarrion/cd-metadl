import torch
import torch.nn as nn
from typing import List
    
    
def update_weights(weights: List[torch.Tensor], 
                   grads: List[torch.Tensor], 
                   grad_clip: int, 
                   lr: float) -> List[torch.Tensor]:
    """ Update the weights based on the gradients and learning rate.

    Args:
        weights (List[torch.Tensor]): Weights to be updated.
        grads (List[torch.Tensor]): Gradients for each weight.
        grad_clip (int): Boundary for clipping the gradients.
        lr (float): Learning rate.

    Returns:
        List[torch.Tensor]: Updated weights
    """
    if grad_clip is not None:
        grads = [torch.clamp(p, -grad_clip, +grad_clip) for p in grads]
    
    new_weights = [weights[i] - lr * grads[i] for i in range(len(grads))]
    
    return new_weights


def get_grads(model: nn.Module, 
              X_train: torch.Tensor, 
              y_train: torch.Tensor, 
              weights: List[torch.Tensor] = None, 
              second_order: bool = False, 
              retain_graph: bool = False) -> List[torch.Tensor]:
    """ Compute the gradients of processing the specified input.

    Args:
        model (nn.Module): Model to be used.
        X_train (torch.Tensor): Support set images.
        y_train (torch.Tensor): Support set labels.
        weights (List[torch.Tensor], optional): Weights to be used by the 
            model. Defaults to None.
        second_order (bool, optional): Boolean flag to control if second order
            derivatives should be computed. Defaults to False.
        retain_graph (bool, optional): Boolean flag to control if the 
            derivation graph should be retained. Defaults to False.

    Returns:
        List[torch.Tensor]: Gradients of the operation
    """
    model.zero_grad()
    if weights is None:
        weights = model.parameters()
        out = model(X_train)
    else:
        out = model.forward_weights(X_train, weights)
    
    loss = model.criterion(out, y_train)
    grads = torch.autograd.grad(loss, weights, create_graph=second_order, 
        retain_graph=retain_graph)
    
    return list(grads)
    