import os
import errno
from collections import namedtuple
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
import os
import torch as T
import contextlib
import numpy as np
from copy import copy
from shared_lib.utils.helpers.model import get_child_attr


def count_params(model):
    """Returns the number of training parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_bpe(bart_dir):
    """Setups BPE encoder corresponding to GPT-2."""
    args_class = namedtuple("Args", ["gpt2_encoder_json", "gpt2_vocab_bpe"])
    args = args_class(gpt2_encoder_json=os.path.join(bart_dir, 'encoder.json'),
                      gpt2_vocab_bpe=os.path.join(bart_dir, 'vocab.bpe'))
    bpe = GPT2BPE(args)
    return bpe


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_params(module):
    for param in module.parameters():
        param.requires_grad = True


def ar_sample(scores, nsamples):
    """Performs auto-regressive sampling by masking previous samples.

    Args:
        scores (np.array): float scores.
        nsamples (int): the number of samples.

    Returns:
        sampled_indxs (list):
    """
    assert isinstance(scores, np.ndarray)
    sampled_indxs = []
    scores = copy(scores)
    for _ in range(nsamples):
        denom = scores.sum(-1, keepdims=True)
        distr = scores / denom

        sel_indx = np.random.choice(range(len(scores)), p=distr)

        scores[sel_indx] = 0
        sampled_indxs.append(sel_indx)

    return sampled_indxs


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = T.arange(max_len, device=length.device, dtype=length.dtype)\
               .expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = T.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def create_flat_prior(prev_sel_indxs, vocab_size):
    """Creates a flat prior distribution where previous selections have 0
    probability.
    """
    blocker_mask = create_blocker_mask(prev_sel_indxs, vocab_size)
    prior = T.ones_like(blocker_mask, dtype=T.float)
    prior = prior.masked_fill_(blocker_mask, 0.)
    prior = prior / prior.sum(-1, keepdim=True)
    return prior


def create_blocker_mask(prev_sel_indxs, vocab_size=None):
    """Creates a mask where previously selected indxs are masked. The first
    selected indxs is usually dummy.

    Args:
        prev_sel_indxs (LongTensor): previously selected indxs of docs.
            (bsz, seq_len)
        vocab_size (int): the number of vocabulary elements.

    Returns:
        blocker_mask (BoolTensor): ``True`` is assigned to blocked vocabulary elems.
            (bsz, seq_len, vocab_size)
    """
    bsz, seq_len = prev_sel_indxs.shape
    if vocab_size is None:
        vocab_size = prev_sel_indxs.max().item() + 1
    blocker_mask = T.zeros((bsz, seq_len, vocab_size),
                           device=prev_sel_indxs.device, dtype=T.long)
    blocker_mask[T.arange(bsz).unsqueeze(-1), T.arange(seq_len), prev_sel_indxs] = 1
    blocker_mask = T.cumsum(blocker_mask, dim=1).bool()
    return blocker_mask


def update_names(model, name, state_dict):
    # updating to new names
    def do_upgrade(m, prefix):
        if len(prefix) > 0:
            prefix += "."
        for n, c in m.named_children():
            name = prefix + n
            if hasattr(c, "upgrade_state_dict_named"):
                c.upgrade_state_dict_named(state_dict, name)
            elif hasattr(c, "upgrade_state_dict"):
                c.upgrade_state_dict(state_dict)
            do_upgrade(c, name)
    do_upgrade(model, name)


def copy_module_params(model, state_dict, module_names):
    """Copied module parameters to the state_dict."""
    for module_name in module_names:
        module = get_child_attr(model, module_name)
        for param_name, param_value in module.named_parameters():
            full_param_name = f'{module_name}.{param_name}'
            if full_param_name not in state_dict:
                state_dict[full_param_name] = copy(param_value)
