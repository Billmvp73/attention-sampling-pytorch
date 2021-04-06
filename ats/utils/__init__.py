"""Provide utility functions to the rest of the modules."""
from functools import partial

import torch
import matplotlib.pyplot as plt

def to_tensor(x, dtype=torch.int32, device=None):
    """If x is a Tensor return it as is otherwise return a constant tensor of
    type dtype."""
    device = torch.device('cpu') if device is None else device
    if torch.is_tensor(x):
        return x.to(device)

    return torch.tensor(x, dtype=dtype, device=device)


def to_dtype(x, dtype):
    """Cast Tensor x to the dtype """
    return x.type(dtype)


to_float16 = partial(to_dtype, dtype=torch.float16)
to_float32 = partial(to_dtype, dtype=torch.float32)
to_float64 = partial(to_dtype, dtype=torch.float64)
to_double = to_float64
to_int8 = partial(to_dtype, dtype=torch.int8)
to_int16 = partial(to_dtype, dtype=torch.int16)
to_int32 = partial(to_dtype, dtype=torch.int32)
to_int64 = partial(to_dtype, dtype=torch.int64)


def expand_many(x, axes):
    """Call expand_dims many times on x once for each item in axes."""
    for ax in axes:
        x = torch.unsqueeze(x, ax)
    return x

def visualize(patch):
    np_patch = patch.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(np_patch)
    plt.show()

def showPatch(patch, img):
    np_patch = patch.cpu().numpy().transpose(1, 2, 0)
    np_img = img.cpu().numpy().transpose(1, 2, 0)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np_patch)
    axs[0].set_title("Patch")
    axs[1].imshow(np_img)
    axs[1].set_title("Image")
    plt.show()
    # visualize(patch)
    # visualize(img)
