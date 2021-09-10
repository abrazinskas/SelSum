from selsum.utils.constants.model import SEL_INDXS, SEL_INDXS_MASK, FEATS, FEATS_MASK,\
    PRIOR_DISTR, PREV_SEL_INDXS
from fairseq.data.data_utils import collate_tokens
from selsum.utils.helpers.model import create_flat_prior
from selsum.utils.helpers.data import pad_feats
import torch as T


def collate_subsampled(samples, batch=None):
    """Collates data after it was subsampled.

    Assumes that it has selected indxs obtained from the posterior or other
    distribution.

    Used in E- and M- steps.
    """
    batch = batch if batch is not None else {}

    if len(samples) == 0:
        return batch

    # features, which are assumed to have the left features for a dummy doc
    feats = [s['feats'] for s in samples]
    feats = T.stack(feats)
    feats_mask = T.stack([s['feats_mask'] for s in samples])

    doc_coll_size = feats.size(1)

    # selected review indices
    if samples[0]['sel_indxs'] is not None:
        sel_indxs = [T.LongTensor(s['sel_indxs']) for s in samples]
        sel_indxs = collate_tokens(sel_indxs, pad_idx=-1)
        sel_indxs_mask = sel_indxs < 0

        # adding the last 'dummy' review features to the first position
        prev_sel_indxs = sel_indxs.clone()
        prev_sel_indxs[:, -1] = doc_coll_size - 1  # the last document is dummy
        prev_sel_indxs = T.roll(prev_sel_indxs, shifts=1, dims=1)
    else:
        sel_indxs = None
        sel_indxs_mask = None
        prev_sel_indxs = None

    # the flat prior distribution
    p_distr = create_flat_prior(prev_sel_indxs, vocab_size=doc_coll_size)

    batch['net_input'][SEL_INDXS] = sel_indxs
    batch['net_input'][SEL_INDXS_MASK] = sel_indxs_mask
    batch['net_input'][FEATS] = feats
    batch['net_input'][FEATS_MASK] = feats_mask
    batch['net_input'][PREV_SEL_INDXS] = prev_sel_indxs
    batch['net_input'][PRIOR_DISTR] = p_distr

    return batch


def collate_features(samples, batch=None, add_dummy=True):
    """Collates features, adds a dummy vector of zeros to indicate a padded
    source document. Used in the dataset loading before subsampling.
    """
    batch = batch if batch is not None else {}
    if len(samples) == 0:
        return batch

    feats = [s['feats'] for s in samples]
    feats, feats_mask = pad_feats(feats)

    if add_dummy:
        # adding a dummy src that is used for at the last position to perform
        # decoding
        feats = T.cat([feats, T.zeros_like((feats[:, 0]).unsqueeze(1))], dim=1)
        feats_mask = T.cat([feats_mask,
                            T.ones((feats_mask.size(0), 1),
                                   device=feats.device).bool()], dim=1)
    if 'net_input' not in batch:
        batch['net_input'] = {}
    batch['net_input'][FEATS] = feats
    batch['net_input'][FEATS_MASK] = feats_mask

    return batch
