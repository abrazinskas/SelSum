from selsum.utils.posterior_generator import PosteriorGenerator
from shared_lib.utils.helpers.general import flatten
from selsum.utils.constants.model import FEATS, FEATS_MASK
from selsum.utils.helpers.collators import collate_subsampled
from selsum.data.abs_dataset import collate as abs_collate
import numpy as np
from copy import copy
from fairseq.utils import apply_to_sample


def subsample(distr_model, sample, rsample_num, pad_indx, eos_indx, mask_idx,
              ndocs=1, sample_type='inf'):
    """Performs sub-sampling by selecting a fixed maximum number of reviews.
    Wraps it back to a batch that can be directly fed to the model.

    Multiple random samples are allocated along the first axis, i.e.,
    (bsz*rsample_num, *)

    Args:
        distr_model: module that yields logits.
        ndocs: how many document ids to sample from the distribution.
    """
    assert sample_type in ['posterior', 'posterior_greedy', 'prior']

    feats = sample['net_input'][FEATS]
    feats_mask = sample['net_input'][FEATS_MASK]
    src = sample['extra']['source']
    tgt = sample['extra']['target']
    id = sample['extra']['id']

    bs, padded_total_docs = feats.shape[:2]

    # repeating to accommodate multiple samples per data-point
    feats = feats.unsqueeze(1).repeat((1, rsample_num, 1, 1)) \
        .view(bs * rsample_num, padded_total_docs, -1)
    feats_mask = feats_mask.unsqueeze(1).repeat((1, rsample_num, 1)) \
        .view(bs * rsample_num, padded_total_docs)
    src = repeat_list_tensors(src, nsamples=rsample_num, by_ref=True)
    tgt = repeat_list_tensors(tgt, nsamples=rsample_num, by_ref=True)
    id = flatten([[i for _ in range(rsample_num)] for i in id])
    rev_counts = (~feats_mask).sum(-1).cpu().numpy()

    if sample_type == 'prior' or max(rev_counts) <= ndocs :
        sel_indxs = sample_from_p(sample_size=ndocs,
                                  total_rev_counts=rev_counts)
    elif sample_type in ['posterior', 'posterior_greedy']:
        greedy = sample_type == 'posterior_greedy'
        feat_sample = {'net_input': {FEATS: feats, FEATS_MASK: feats_mask}}
        sel_indxs, _ = sample_from_q(model=distr_model, sample=feat_sample,
                                     bos_idx=-1, pad_idx=-1,
                                     sample_size=ndocs, greedy=greedy)
        sel_indxs = sel_indxs.cpu().numpy()
    else:
        raise NotImplementedError

    assert len(feats) == len(feats_mask) == len(src)

    coll = []

    for indx in range(len(sel_indxs)):
        _id = id[indx]
        _feats = feats[indx]
        _feats_mask = feats_mask[indx]
        _sel_indxs = sel_indxs[indx]
        _src = src[indx]
        _tgt = tgt[indx] if tgt is not None else None
        _ndocs = rev_counts[indx].item()

        # removing all padded sampled documents because the number of
        # selected documents can't exceed the number of documents in the
        # collection
        _sel_indxs = _sel_indxs[:_ndocs]

        assert isinstance(_src, list)
        _subs_src = [_src[i] for i in _sel_indxs]

        # storing to the collector
        coll.append({'source': _subs_src, 'target': _tgt,
                     'sel_indxs': _sel_indxs, 'id': _id, 'feats': _feats,
                     'feats_mask': _feats_mask})

    new_sample = abs_collate(coll, pad_idx=pad_indx, eos_idx=eos_indx,
                             mask_idx=mask_idx)
    new_sample = collate_subsampled(coll, new_sample)
    new_sample = apply_to_sample(lambda tensor: tensor.to(feats.device),
                                 new_sample)
    return new_sample


def sample_from_q(model, sample, sample_size, bos_idx=-1, pad_idx=-1,
                  greedy=False, temperature=1.):
    """Auto-regressively samples from the approximate posterior `sample_size` times.

    Args:
        model: posterior model.
        sample (dict): containing 'net_input' with features and mask.
        sample_size (int): the number of times to sample from the posterior.
        temperature (float): used in sampling to re-scale scores.

    """
    seq_sampler = PosteriorGenerator(model=model, pad_idx=pad_idx,
                                     bos_idx=bos_idx, greedy=greedy,
                                     temperature=temperature)
    doc_indxs, probs = seq_sampler(sample=sample, max_seq_len=sample_size)
    return doc_indxs, probs


def sample_from_p(sample_size, total_rev_counts):
    """Samples from the flat categorical distribution without replacement.

    Args:
        sample_size (int): the number of samples to yield for each data-point.
        total_rev_counts (list): total number of reviews per each data-point.
    """
    samples = []
    for rc in total_rev_counts:
        if rc > sample_size:
            sample = np.random.choice(rc, replace=False, size=sample_size)
        else:
            sample = np.arange(rc)
        samples.append(sample)
    return samples


def repeat_list_tensors(lst_tens, nsamples, by_ref=False):
    """Repeats a list of tensors either by copying or by reference."""
    res = []
    for l in lst_tens:
        res += [l if by_ref else copy(l) for _ in range(nsamples)]
    return res


def create_flat_distr(support_size, sel_indxs):
    """Creates a flat categorical distribution without replacement."""
    flat_distr = [np.ones((support_size,)) / support_size
                  for i in range(len(sel_indxs))]
    return flat_distr
