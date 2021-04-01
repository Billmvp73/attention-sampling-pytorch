import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
from ..data.from_tensors import FromTensors, FromMultiTensors
from .sampling import sample, multisample
from .expectation import Expectation
from ..utils.layers import SampleSoftmax

class SamplePatches(nn.Module):
    """SamplePatches samples from a high resolution image using an attention
    map. The layer expects the following inputs when called `x_low`, `x_high`,
    `attention`. `x_low` corresponds to the low resolution view of the image
    which is used to derive the mapping from low resolution to high. `x_high`
    is the tensor from which we extract patches. `attention` is an attention
    map that is computed from `x_low`.
    Arguments
    ---------
        n_patches: int, how many patches should be sampled
        patch_size: int, the size of the patches to be sampled (squared)
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.
        replace: bool, whether we should sample with replacement or without
        use_logits: bool, whether of not logits are used in the attention map
    """

    def __init__(self, n_patches, patch_size, receptive_field=0, replace=False,
                 use_logits=False, **kwargs):
        self._n_patches = n_patches
        self._patch_size = (patch_size, patch_size)
        self._receptive_field = receptive_field
        self._replace = replace
        self._use_logits = use_logits

        super(SamplePatches, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Legacy function of the pytorch implementation """
        shape_low, shape_high, shape_att = input_shape

        # Figure out the shape of the patches
        patch_shape = (shape_high[1], *self._patch_size)

        patches_shape = (shape_high[0], self._n_patches, *patch_shape)

        # Sampled attention
        att_shape = (shape_high[0], self._n_patches)

        return [patches_shape, att_shape]

    def forward(self, x_low, x_high, attention):
        sample_space = attention.shape[1:]
        samples, sampled_attention = sample(
            self._n_patches,
            attention,
            sample_space,
            replace=self._replace,
            use_logits=self._use_logits
        )

        offsets = torch.zeros_like(samples).float()
        if self._receptive_field > 0:
            offsets = offsets + self._receptive_field / 2

        # Get the patches from the high resolution data
        # Make sure that below works
        x_low = x_low.permute(0, 2, 3, 1)
        x_high = x_high.permute(0, 2, 3, 1)
        assert x_low.shape[-1] == x_high.shape[-1], "Channels should be last for now"
        patches, _ = FromTensors([x_low, x_high], None).patches(
            samples,
            offsets,
            sample_space,
            torch.Tensor([x_low.shape[1:-1]]).view(-1) - self._receptive_field,
            self._patch_size,
            0,
            1
        )

        return [patches, sampled_attention]


class ATSModel(nn.Module):
    """ Attention sampling model that perform the entire process of calculating the
        attention map, sampling the patches, calculating the features of the patches,
        the expectation and classifices the features.
        Arguments
        ---------
        attention_model: pytorch model, that calculated the attention map given a low
                         resolution input image
        feature_model: pytorch model, that takes the patches and calculated features
                       of the patches
        classifier: pytorch model, that can do a classification into the number of
                    classes for the specific problem
        n_patches: int, the number of patches to sample
        patch_size: int, the patch size (squared)
        receptive_field: int, how large is the receptive field of the attention network.
                         It is used to map the attention to high resolution patches.
        replace: bool, if to sample with our without replacment
        use_logts: bool, if to use logits when sampling
    """

    def __init__(self, attention_model, feature_model, classifier, n_patches, patch_size, receptive_field=0,
                 replace=False, use_logits=False):
        super(ATSModel, self).__init__()

        self.attention_model = attention_model
        self.feature_model = feature_model
        self.classifier = classifier

        self.sampler = SamplePatches(n_patches, patch_size, receptive_field, replace, use_logits)
        self.expectation = Expectation(replace=replace)

        self.patch_size = patch_size
        self.n_patches = n_patches

    def forward(self, x_low, x_high):
        # First we compute our attention map
        attention_map = self.attention_model(x_low)

        # Then we sample patches based on the attention
        patches, sampled_attention = self.sampler(x_low, x_high, attention_map)

        # We compute the features of the sampled patches
        channels = patches.shape[2]
        patches_flat = patches.view(-1, channels, self.patch_size, self.patch_size)
        patch_features = self.feature_model(patches_flat)
        dims = patch_features.shape[-1]
        patch_features = patch_features.view(-1, self.n_patches, dims)

        sample_features = self.expectation(patch_features, sampled_attention)

        y = self.classifier(sample_features)

        return y, attention_map, patches, x_low

#---------- Start Processing MultiResolution Images -----

class MultiSamplePatches(nn.Module):
    """SamplePatches samples from a high resolution image using an attention
    map. The layer expects the following inputs when called `x_low`, `x_high`,
    `attention`. `x_low` corresponds to the low resolution view of the image
    which is used to derive the mapping from low resolution to high. `x_high`
    is the tensor from which we extract patches. `attention` is an attention
    map that is computed from `x_low`.
    Arguments
    ---------
        n_patches: int, how many patches should be sampled
        patch_size: int, the size of the patches to be sampled (squared)
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.
        replace: bool, whether we should sample with replacement or without
        use_logits: bool, whether of not logits are used in the attention map
    """

    def __init__(self, n_patches, patch_size, scales=[1], receptive_field=0, replace=False,
                 use_logits=False, **kwargs):
        self._n_patches = n_patches
        self._patch_size = (patch_size, patch_size)
        self._receptive_field = receptive_field
        self._replace = replace
        self._use_logits = use_logits
        self.scales = scales

        super(MultiSamplePatches, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Legacy function of the pytorch implementation """
        shape_low, shape_high, shape_att = input_shape

        # Figure out the shape of the patches
        patch_shape = (shape_high[1], *self._patch_size)

        patches_shape = (shape_high[0], self._n_patches, *patch_shape)

        # Sampled attention
        att_shape = (shape_high[0], self._n_patches)

        return [patches_shape, att_shape]

    def forward(self, x_lows, x_highs, attention, map_index):
        sample_space = attention.shape[1:]
        samples, sampled_attention, samples_index = multisample(
            self._n_patches,
            attention,
            map_index,
            sample_space,
            replace=self._replace,
            use_logits=self._use_logits
        )

        offsets = torch.zeros_like(samples).float()
        if self._receptive_field > 0:
            offsets = offsets + self._receptive_field / 2

        # Get the patches from the high resolution data
        # Make sure that below works
        permute_lows = []
        permute_highs = []
        for x_low, x_high in zip(x_lows, x_highs):
            permute_lows.append(x_low.permute(0, 2, 3, 1))
            permute_highs.append(x_high.permute(0, 2, 3, 1))
        x_lows = permute_lows
        x_highs = permute_highs
        # x_low = x_low.permute(0, 2, 3, 1)
        # x_high = x_high.permute(0, 2, 3, 1)
        for x_low, x_high in zip(x_lows, x_highs):
            assert x_low.shape[-1] == x_high.shape[-1], "Channels should be last for now"

        # for i, (x_low, x_high, scale) in enumerate(zip(x_lows, x_highs, self.scales)):
        #     equal_index = torch.where(samples_index==i)
        #     samples_i = samples[equal_index]
            
        patches, _ = FromMultiTensors([x_lows, x_highs], None, self.scales).patches(
            samples,
            samples_index,
            offsets,
            sample_space,
            [torch.Tensor([x_low.shape[1:-1]]).view(-1) - self._receptive_field for x_low in x_lows],
            self._patch_size,
            0,
            1
        )

        return [patches, sampled_attention]

class MultiATSModel(nn.Module):
    """ Attention sampling model that perform the entire process of calculating the
        attention map, sampling the patches, calculating the features of the patches,
        the expectation and classifices the features.
        Arguments
        ---------
        attention_model: pytorch model, that calculated the attention map given a low
                         resolution input image
        feature_model: pytorch model, that takes the patches and calculated features
                       of the patches
        classifier: pytorch model, that can do a classification into the number of
                    classes for the specific problem
        n_patches: int, the number of patches to sample
        patch_size: int, the patch size (squared)
        receptive_field: int, how large is the receptive field of the attention network.
                         It is used to map the attention to high resolution patches.
        replace: bool, if to sample with our without replacment
        use_logts: bool, if to use logits when sampling
    """

    def __init__(self, attention_model, feature_model, classifier, n_patches, patch_size, scales,receptive_field=0,
                 replace=False, use_logits=False):
        super(MultiATSModel, self).__init__()

        self.attention_model = attention_model
        self.sampleSoftmax = SampleSoftmax(True, 1e-4)
        self.feature_model = feature_model
        self.classifier = classifier

        self.multiSampler = MultiSamplePatches(n_patches, patch_size, scales, receptive_field, replace, use_logits)
        self.sampler = SamplePatches(n_patches, patch_size, receptive_field, replace, use_logits)
        self.expectation = Expectation(replace=replace)

        self.patch_size = patch_size
        self.n_patches = n_patches

        self.scales = scales
        assert self.scales[0] == 1


    def forward(self, x_lows, x_highs):
        high_ats_shape = None
        attention_maps = []
        for i, (x_low, x_high, scale) in enumerate(zip(x_lows, x_highs,self.scales)):

            # First we compute our attention map
            attention_map = self.attention_model(x_low)
            attention_map = attention_map.squeeze(1)
            attention_maps.append(attention_map)
            if scale == 1:
                high_ats_shape = attention_map.shape
            #     upsampled_map = attention_map
            # # Option1: upsample downsampled attention map
            # else:
            #     attention_map = torch.unsqueeze(attention_map, dim=1)
            #     upsampled_map = F.interpolate(attention_map, size=(high_ats_shape[-2], high_ats_shape[-1]), mode="nearest")
            #     upsampled_map = torch.squeeze(upsampled_map)
            #     # TBD: do we need to normalize the upsampled attention map?
            #     # total_weights = torch.sum(upsampled_map.view(upsampled_map.shape[0], -1), dim=1)
            #     # upsampled_map = torch.matmul(upsampled_map.view(upsampled_map.shape[0], -1), 1 / total_weights)
            # attention_maps.append(upsampled_map)
        final_map = torch.zeros_like(attention_maps[0])
        final_index = torch.zeros(high_ats_shape, dtype=torch.int32, device=attention_maps[0].device)
        for x in range(high_ats_shape[1]):
            for y in range(high_ats_shape[2]):
                weights_xy = []
                for ats_map, scale in zip(attention_maps, self.scales):
                    sx = min(int(x * scale), ats_map.shape[1] - 1)
                    sy = min(int(y * scale), ats_map.shape[2] - 1)
                    if sx != x * scale or sy != y * scale:
                        empty = torch.zeros_like(attention_maps[0][:, 0, 0])
                        empty[:] = -float('inf')
                        weights_xy.append(empty)
                    else:
                        weights_xy.append(ats_map[:, sx, sy])
                merged_xy = torch.stack(weights_xy)
                max_xy, max_index = torch.max(merged_xy, dim=0)
                final_map[:, x, y] = max_xy
                final_index[:, x, y] = max_index
        # final_map = torch.normalize(final_map, )
        # reshape_map = final_map.view((final_map.shape[0], -1))
        # sum_map = torch.sum(reshape_map, dim=1).view(final_map.shape[0], 1)
        # final_map = (reshape_map / sum_map).view((high_ats_shape[0], high_ats_shape[1], high_ats_shape[2]))
        final_map = self.sampleSoftmax(final_map.unsqueeze(1))
        patches, sampled_attention = self.multiSampler(x_lows, x_highs, final_map, final_index)


        # We compute the features of the sampled patches
        channels = patches.shape[2]
        patches_flat = patches.view(-1, channels, self.patch_size, self.patch_size)
        patch_features = self.feature_model(patches_flat)
        dims = patch_features.shape[-1]
        patch_features = patch_features.view(-1, self.n_patches, dims)

        sample_features = self.expectation(patch_features, sampled_attention)

        y = self.classifier(sample_features)

        return y, final_map, patches, x_low
    
class MultiParallelATSModel(nn.Module):
    """ Attention sampling model that perform the entire process of calculating the
        attention map, sampling the patches, calculating the features of the patches,
        the expectation and classifices the features.
        Arguments
        ---------
        attention_model: pytorch model, that calculated the attention map given a low
                         resolution input image
        feature_model: pytorch model, that takes the patches and calculated features
                       of the patches
        classifier: pytorch model, that can do a classification into the number of
                    classes for the specific problem
        n_patches: int, the number of patches to sample
        patch_size: int, the patch size (squared)
        receptive_field: int, how large is the receptive field of the attention network.
                         It is used to map the attention to high resolution patches.
        replace: bool, if to sample with our without replacment
        use_logts: bool, if to use logits when sampling
    """

    def __init__(self, attention_model, feature_model, classifier, n_patches, patch_size, scales,receptive_field=0,
                 replace=False, use_logits=False):
        super(MultiParallelATSModel, self).__init__()

        self.attention_model = attention_model
        self.feature_model = feature_model
        self.classifier = classifier

        self.multiSampler = MultiSamplePatches(n_patches, patch_size, scales, receptive_field, replace, use_logits)
        self.sampler = SamplePatches(n_patches, patch_size, receptive_field, replace, use_logits)
        self.expectation = Expectation(replace=replace)

        self.patch_size = patch_size
        self.n_patches = n_patches

        self.scales = scales
        assert self.scales[0] == 1


    def forward(self, x_lows, x_highs):
        high_ats_shape = None
        attention_maps = []
        multi_patches = []
        multi_sampled_attention = []
        for i, (x_low, x_high, scale) in enumerate(zip(x_lows, x_highs,self.scales)):

            # First we compute our attention map
            attention_map = self.attention_model(x_low)
            
            patches, sampled_attention = self.sampler(x_low, x_high, attention_map)
            # patches, sampled_attention = self.multiSampler(x_lows, x_highs, norm_map, max_index)
            multi_patches.append(patches)
            multi_sampled_attention.append(sampled_attention)
            attention_maps.append(attention_map)
        patches = torch.cat(multi_patches, 1)
        sampled_attention = torch.cat(multi_sampled_attention, 1)
        # We compute the features of the sampled patches
        channels = patches.shape[2]
        patches_flat = patches.view(-1, channels, self.patch_size, self.patch_size)
        patch_features = self.feature_model(patches_flat)
        dims = patch_features.shape[-1]
        patch_features = patch_features.view(-1, self.n_patches * len(self.scales), dims)

        sample_features = self.expectation(patch_features, sampled_attention / len(self.scales))

        y = self.classifier(sample_features)

        return y, attention_maps, patches, x_lows