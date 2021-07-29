import numpy as np
import torch
from sklearn.metrics import accuracy_score


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, list):
        return [move_to(v, device) for v in var]
    elif isinstance(var, tuple):
        return (move_to(v, device) for v in var)
    return var.to(device)


def calc_cls_measures(probs, label):
    """Calculate multi-class classification measures (Accuracy)
    :probs: NxC numpy array storing probabilities for each case
    :label: ground truth label
    :returns: a dictionary of accuracy
    """
    label = label.reshape(-1, 1)
    n_classes = probs.shape[1]
    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(label, preds)

    metric_collects = {'accuracy': accuracy}
    return metric_collects

from torch._six import inf


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    filter_params = []
    for param in parameters:
        filter_params += list(filter(lambda p: p.grad is not None, param))
    # parameters = list(filter(lambda p: p.grad is not None, parameters))
    # parameters = filter_params
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in filter_params)
    else:
        total_norm = 0
        for p in filter_params:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in filter_params:
            p.grad.data.mul_(clip_coef)
    return total_norm
