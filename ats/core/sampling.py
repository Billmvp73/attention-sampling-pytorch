"""Implement sampling from a multinomial distribution on a n-dimensional
tensor."""
import torch
import torch.distributions as dist


def _sample_with_replacement(logits, n_samples):
    """Sample with replacement using the pytorch categorical distribution op."""
    distribution = dist.categorical.Categorical(logits=logits)
    return distribution.sample(sample_shape=torch.Size([n_samples])).transpose(0, 1)


def _sample_without_replacement(logits, n_samples):
    """Sample without replacement using the Gumbel-max trick.
    See lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
    """
    z = -torch.log(-torch.log(torch.rand_like(logits)))
    return torch.topk(logits+z, k=n_samples)[1]


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return torch.stack(tuple(reversed(out)))

def norm_resample(n_samples, multi_samples, multi_attention, scales, norm_atts_weight=False, replace = False, use_logits=False):
    """Sample the top k ones from the previously sampled k patches per scale.

    n sampled patches for one scale, n*k in total, (batch_size, n_samples, k, n_dims)
    multi_sampled_attentions (b, n, k)
    """
    # if use_logits:
    #     multi_logits = multi_attention
    # else:
    #     multi_logits = [torch.log(attention) for attention in multi_attention]
    
    # sampling_function = (_sample_with_replacement if replace else _sample_without_replacement)
    
    """
    First normalize the sampled attention.
    """
    norm_atts = []
    # scale[i] is the ratio of the number of patches at scale 1 to the number of patches at scale s[i]
    for i, scale in enumerate(scales):
        norm_att_i = multi_attention[i]*scale
        norm_atts.append(norm_att_i)
    # norm_atts = torch.stack(norm_atts)
    norm_atts = torch.cat(norm_atts, 1)
    unnorm_atts = torch.cat(multi_attention, 1)
    top_atts, top_ind = torch.topk(norm_atts, n_samples, 1)
    unnorm_top_atts = torch.gather(unnorm_atts, 1, top_ind)
    sampled_scales = top_ind // n_samples
    batch_size = top_atts.shape[0]
    total_samples = torch.cat(multi_samples, 1)
    total_num, c, s0, s1 = total_samples.shape[1:]

    total_samples = total_samples.reshape(batch_size, total_num, -1)
    #gather samples given topk indices
    expand_indices = top_ind.unsqueeze(2).expand(-1, -1, total_samples.shape[-1])
    top_samples = torch.gather(total_samples, 1, expand_indices).reshape(batch_size, n_samples, c, s0, s1)

    # top_samples = multi_samples[top_ind]
    if norm_atts_weight:
        return top_samples, top_atts, unnorm_top_atts, sampled_scales # TODO: why the loss could be negative? The attentions associated with the patches are too small?
    return top_samples, unnorm_top_atts, unnorm_top_atts, sampled_scales
    # Flatten the attention distribution and sampel from it
    

def sample(n_samples, attention, sample_space, replace=False,
           use_logits=False):
    """Sample from the passed in attention distribution.
    Arguments
    ---------
    n_samples: int, the number of samples per datapoint
    attention: tensor, the attention distribution per datapoint (could be logits
               or normalized)
    sample_space: This should always equal K.shape(attention)[1:]
    replace: bool, sample with replacement if set to True (defaults to False)
    use_logits: bool, assume the input is logits if set to True (defaults to False)
    """
    # Make sure we have logits and choose replacement or not
    logits = attention if use_logits else torch.log(attention)
    sampling_function = (
        _sample_with_replacement if replace
        else _sample_without_replacement
    )

    # Flatten the attention distribution and sample from it
    logits = logits.reshape(-1, sample_space[0]*sample_space[1])
    samples = sampling_function(logits, n_samples)

    # Unravel the indices into sample_space
    batch_size = attention.shape[0]
    n_dims = len(sample_space)

    # Gather the attention
    attention = attention.view(batch_size, 1, -1).expand(batch_size, n_samples, -1)
    sampled_attention = torch.gather(attention, -1, samples[:, :, None])[:, :, 0]

    samples = unravel_index(samples.reshape(-1, ), sample_space)
    samples = torch.reshape(samples.transpose(1, 0), (batch_size, n_samples, n_dims))

    return samples, sampled_attention

def multisample(n_samples, attention, map_index, sample_space, replace=False,
           use_logits=False):
    """Sample from the passed in attention distribution.
    Arguments
    ---------
    n_samples: int, the number of samples per datapoint
    attention: tensor, the attention distribution per datapoint (could be logits
               or normalized)
    sample_space: This should always equal K.shape(attention)[1:]
    replace: bool, sample with replacement if set to True (defaults to False)
    use_logits: bool, assume the input is logits if set to True (defaults to False)
    """
    # Make sure we have logits and choose replacement or not
    logits = attention if use_logits else torch.log(attention)
    sampling_function = (
        _sample_with_replacement if replace
        else _sample_without_replacement
    )

    # Flatten the attention distribution and sample from it
    logits = logits.reshape(-1, sample_space[0]*sample_space[1])
    samples = sampling_function(logits, n_samples)

    # Unravel the indices into sample_space
    batch_size = attention.shape[0]
    n_dims = len(sample_space)

    # Gather the attention
    attention = attention.view(batch_size, 1, -1).expand(batch_size, n_samples, -1)
    sampled_attention = torch.gather(attention, -1, samples[:, :, None])[:, :, 0]

    # Untested: Gather the attention index
    map_index = map_index.view(batch_size, 1, -1).expand(batch_size, n_samples, -1)
    sampled_ats_index = torch.gather(map_index, -1, samples[:, :, None])[:, :, 0]

    samples = unravel_index(samples.reshape(-1, ), sample_space)
    samples = torch.reshape(samples.transpose(1, 0), (batch_size, n_samples, n_dims))

    return samples, sampled_attention, sampled_ats_index
