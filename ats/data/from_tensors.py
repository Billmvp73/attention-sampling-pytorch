import os
import torch
from joblib import Parallel, delayed

from ..utils import to_tensor, to_float32, to_int32, expand_many


def _extract_patch(img_b, coord, patch_size):
    """ Extract a single patch """
    x_start = int(coord[0])
    x_end = x_start + int(patch_size[0])
    y_start = int(coord[1])
    y_end = y_start + int(patch_size[1])

    patch = img_b[:, x_start:x_end, y_start:y_end]
    return patch


def _extract_patches_batch(b, img, offsets, patch_size, num_patches, extract_patch_parallel=False):
    """ Extract patches for a single batch. This function can be called in a for loop or in parallel.
        This functions returns a tensor of patches of size [num_patches, channels, width, height] """
    patches = []

    # Extracting in parallel is more expensive than doing it sequentially. This I left it in here
    if extract_patch_parallel:
        num_jobs = min(os.cpu_count(), num_patches)
        patches = Parallel(n_jobs=num_jobs)(
            delayed(_extract_patch)(img[b], offsets[b, p], patch_size) for p in range(num_patches))

    else:
        # Run extraction sequentially
        for p in range(num_patches):
            patch = _extract_patch(img[b], offsets[b, p], patch_size)
            patches.append(patch)

    return torch.stack(patches)


def extract_patches(img, offsets, patch_size, extract_batch_parallel=False):
    img = img.permute(0, 3, 1, 2)

    num_patches = offsets.shape[1]
    batch_size = img.shape[0]

    # I pad the images with zeros for the cases that a part of the patch falls outside the image
    pad_const = int(patch_size[0].item() / 2)
    pad_func = torch.nn.ConstantPad2d(pad_const, 0.0)
    img = pad_func(img)

    # Add the pad_const to the offsets, because everything is now shifted by pad_const
    offsets = offsets + pad_const

    all_patches = []

    # Extracting in parallel is more expensive than doing it sequentially. This I left it in here
    if extract_batch_parallel:
        num_jobs = min(os.cpu_count(), batch_size)
        all_patches = Parallel(n_jobs=num_jobs)(
            delayed(_extract_patches_batch)(b, img, offsets, patch_size, num_patches) for b in range(batch_size))

    else:
        # Run sequentially over the elements in the batch
        for b in range(batch_size):
            patches = _extract_patches_batch(b, img, offsets, patch_size, num_patches)
            all_patches.append(patches)

    return torch.stack(all_patches)


class FromTensors:
    def __init__(self, xs, y):
        """Given input tensors for each level of resolution provide the patches.
        Arguments
        ---------
        xs: list of tensors, one tensor per resolution in ascending
            resolutions, namely the lowest resolution is 0 and the highest
            is len(xs)-1
        y: tensor or list of tensors or None, the targets can be anything
           since it is simply returned as is
        """
        self._xs = xs
        self._y = y

    def targets(self):
        # Since the xs were also given to us the y is also given to us
        return self._y

    def inputs(self):
        # We leave it to the caller to add xs and y to the input list if they
        # are placeholders
        return []

    def patches(self, samples, offsets, sample_space, previous_patch_size,
                patch_size, fromlevel, tolevel):
        device = samples.device

        # Make sure everything is a tensor
        sample_space = to_tensor(sample_space, device=device)
        previous_patch_size = to_tensor(previous_patch_size, device=device)
        patch_size = to_tensor(patch_size, device=device)
        shape_from = self._shape(fromlevel)
        shape_to = self._shape(tolevel)

        # Compute the scales
        scale_samples = self._scale(sample_space, tolevel).to(device)
        scale_offsets = self._scale(shape_from, shape_to).to(device)

        # Steps is the offset per pixel of the sample space. Pixel zero should
        # be at position steps/2 and the last pixel should be at
        # space_available - steps/2.
        space_available = to_float32(previous_patch_size) * scale_offsets
        steps = space_available / to_float32(sample_space)

        # Compute the patch start which are also the offsets to be returned
        offsets = to_int32(torch.round(
            to_float32(offsets) * expand_many(scale_offsets, [0, 0]) +
            to_float32(samples) * expand_many(steps, [0, 0]) +
            expand_many(steps / 2, [0, 0]) -
            expand_many(to_float32(patch_size) / 2, [0, 0])
        ))

        # Extract the patches
        patches = extract_patches(
            self._xs[tolevel],
            offsets,
            patch_size
        )

        return patches, offsets

    def data(self, level):
        return self._xs[level]

    def _scale(self, shape_from, shape_to):
        # Compute the tensor that needs to be multiplied with `shape_from` to
        # get `shape_to`
        shape_from = to_float32(to_tensor(shape_from))
        shape_to = to_float32(to_tensor(shape_to))

        return shape_to / shape_from

    def _shape(self, level):
        x = self._xs[level]
        int_shape = x.shape[1:-1]
        if not any(s is None for s in int_shape):
            return int_shape

        return x.shape[1:-1]

# --- Start Processing MultiResolution ---

def _extract_multi_patch(img_b, coord, patch_size):
    """ Extract a single patch """
    x_start = int(coord[0])
    x_end = x_start + int(patch_size[0])
    y_start = int(coord[1])
    y_end = y_start + int(patch_size[1])

    patch = img_b[:, x_start:x_end, y_start:y_end]
    return patch


def _extract_multi_patches_batch(b, imgs, offsets, patch_size, num_patches, samples_index, scales, extract_patch_parallel=False):
    """ Extract patches for a single batch. This function can be called in a for loop or in parallel.
        This functions returns a tensor of patches of size [num_patches, channels, width, height] """
    patches = []

    # Extracting in parallel is more expensive than doing it sequentially. This I left it in here
    if extract_patch_parallel:
        num_jobs = min(os.cpu_count(), num_patches)
        patches = Parallel(n_jobs=num_jobs)(
            delayed(_extract_multi_patch)(imgs[samples_index[b, p]][b], offsets[b, p], patch_size) for p in range(num_patches))

    else:
        # Run extraction sequentially
        for p in range(num_patches):
            s = samples_index[b, p]
            patch = _extract_multi_patch(imgs[s][b], offsets[b, p], patch_size)
            patches.append(patch)

    return torch.stack(patches)

def extract_multi_patches(imgs, offsets, patch_size, samples_index, scales, extract_batch_parallel=False):
    permuted_imgs = []
    for img in imgs:
        permuted_imgs.append(img.permute(0, 3, 1, 2))
    imgs = permuted_imgs

    num_patches = offsets.shape[1]
    batch_size = imgs[0].shape[0]

    # I pad the images with zeros for the cases that a part of the patch falls outside the image
    pad_const = int(patch_size[0].item() / 2)
    pad_func = torch.nn.ConstantPad2d(pad_const, 0.0)
    pad_imgs = []
    for img in imgs:
        pad_imgs.append(pad_func(img))
    # img = pad_func(img)
    imgs = pad_imgs
    # Add the pad_const to the offsets, because everything is now shifted by pad_const
    offsets = offsets + pad_const

    all_patches = []

    # Extracting in parallel is more expensive than doing it sequentially. This I left it in here
    if extract_batch_parallel:
        num_jobs = min(os.cpu_count(), batch_size)
        all_patches = Parallel(n_jobs=num_jobs)(
            delayed(_extract_multi_patches_batch)(b, img, offsets, patch_size, num_patches, samples_index, scales) for b in range(batch_size))

    else:
        # Run sequentially over the elements in the batch
        for b in range(batch_size):
            patches = _extract_multi_patches_batch(b, imgs, offsets, patch_size, num_patches, samples_index, scales)
            all_patches.append(patches)

    return torch.stack(all_patches)

class FromMultiTensors:
    def __init__(self, xs, y, scales):
        """Given input tensors for each level of resolution provide the patches.
        Arguments
        ---------
        xs: list of tensors, one tensor per resolution in ascending
            resolutions, namely the lowest resolution is 0 and the highest
            is len(xs)-1
        y: tensor or list of tensors or None, the targets can be anything
           since it is simply returned as is
        """
        self._xs = xs
        self._y = y
        self._scales = scales

    def targets(self):
        # Since the xs were also given to us the y is also given to us
        return self._y

    def inputs(self):
        # We leave it to the caller to add xs and y to the input list if they
        # are placeholders
        return []

    def patches(self, samples, samples_index, offsets, sample_space, previous_patch_sizes,
                patch_size, fromlevel, tolevel):
        device = samples.device

        # Make sure everything is a tensor
        sample_space = to_tensor(sample_space, device=device)
        patch_size = to_tensor(patch_size, device=device)
        # shape_from = self._shape(fromlevel)
        # shape_to = self._shape(tolevel)
        prev_patch_sizes = []
        for previous_patch_size in previous_patch_sizes:
            prev_patch_size = to_tensor(previous_patch_size, device=device)
            prev_patch_sizes.append(prev_patch_size)
        previous_patch_sizes = prev_patch_sizes
        # previous_patch_sizes = to_tensor(previous_patch_sizes, device=device)

        # Compute the scales
        scales_samples = []
        scales_offsets = []
        computed_offsets = torch.zeros_like(offsets)
        float_samples = to_float32(samples)
        float_offsets = to_float32(offsets)
        for i in range(len(self._scales)):
            shape_from = self._shape(fromlevel, i)
            shape_to = self._shape(tolevel, i)
            scale_samples = self._scale(sample_space, tolevel).to(device)
            scale_offsets = self._scale(shape_from, shape_to).to(device)

            # Steps is the offset per pixel of the sample space. Pixel zero should
            # be at position steps/2 and the last pixel should be at
            # space_available - steps/2.
            space_available = to_float32(previous_patch_sizes[i]) * scale_offsets
            steps = space_available / to_float32(sample_space)

            idx_i = torch.where(samples_index == i)
            # offset_i = float_offsets * expand_many(scale_offsets, [0, 0]) + float_samples * expand_many(steps, [0, 0]) + expand_many(steps / 2, [0, 0]) - expand_many(to_float32(patch_size) / 2, [0, 0])
            
            computed_offsets[idx_i] = float_offsets[idx_i] + expand_many(scale_offsets, [0]) + float_samples[idx_i] * expand_many(steps, [0]) + expand_many(steps/2, [0]) - expand_many(to_float32(patch_size)/2, [0])
        # Compute the patch start which are also the offsets to be returned
        # offsets = to_int32(torch.round(
        #     to_float32(offsets) * expand_many(scale_offsets, [0, 0]) +
        #     to_float32(samples) * expand_many(steps, [0, 0]) +
        #     expand_many(steps / 2, [0, 0]) -
        #     expand_many(to_float32(patch_size) / 2, [0, 0])
        # ))
        offsets = to_int32(torch.round(computed_offsets))

        # Extract the patches
        patches = extract_multi_patches(
            self._xs[tolevel],
            offsets,
            patch_size,
            samples_index,
            self._scales
        )
        # patches = extract_patches(
        #     self._xs[tolevel],
        #     offsets,
        #     patch_size
        # )

        return patches, offsets

    def data(self, level):
        return self._xs[level]

    def _scale(self, shape_from, shape_to):
        # Compute the tensor that needs to be multiplied with `shape_from` to
        # get `shape_to`
        shape_from = to_float32(to_tensor(shape_from))
        shape_to = to_float32(to_tensor(shape_to))

        return shape_to / shape_from

    def _shape(self, level, index):
        x = self._xs[level][index]
        int_shape = x.shape[1:-1]
        if not any(s is None for s in int_shape):
            return int_shape

        return x.shape[1:-1]