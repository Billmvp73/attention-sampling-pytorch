"""Provide utility functions to the rest of the modules."""
from functools import partial
import seaborn as sb
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

# def visualizeGrid(patch, axs, i, j):
#     np_patch = patch.cpu().numpy().transpose(1, 2, 0)
#     axs[i, j].imshow(np_patch)
#     axs[i, j].set_title("Patch(%d, %d)"%(i, j))

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

def patchGrid(patches, maps, imgs, size, label, sampled_scales=None):
    row, col = size
    figs, axs = plt.subplots(row+2, col)
    for i in range(patches.shape[0]):
        r = i // col
        c = i % col
        np_patch = patches[i].cpu().numpy().transpose(1, 2, 0)
        axs[r, c].imshow(np_patch)
        if sampled_scales is None:
            axs[r, c].set_title("(%d)"%(label[i]))
        else:
             axs[r, c].set_title("(%d, %d)"%(sampled_scales[i], label[i]))
        axs[r, c].axis("off")
    for i in range(len(maps)):
        map = maps[i]
        img = imgs[i]
        axs[-2, i].imshow(map)
        axs[-2, i].set_title("Map(%d)"%(i))
        axs[-2, i].axis("off")
        axs[-1, i].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axs[-1, i].set_title("Low(%d)"%(i))
        axs[-1, i].axis("off")
    # plt.colorbar()
    plt.show()
    plt.clf()
    plt.close()

def mapGrid(maps, imgs, scales):
    figs, axs = plt.subplots(2, len(scales))
    for i, (map, img, scale) in enumerate(zip(maps, imgs, scales)):
        axs[0, i].imshow(map.cpu().numpy())
        axs[1, i].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axs[0, i].axis("off")
        axs[1, i].axis("off")
    plt.show()
    plt.clf()
    plt.close()