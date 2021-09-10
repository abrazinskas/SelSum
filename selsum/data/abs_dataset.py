import numpy as np
import torch
from selsum.utils.helpers.data import numel
from fairseq.data import FairseqDataset
from fairseq.data.data_utils import collate_tokens


def flatten_src(source, mask_idx):
    """Flattens source documents to one list. Retains group src indxs.

    Adds a fully masked review at the end that can be selected by padded
    group_indxs.
    """
    group_src_indxs = []
    src_coll = []
    max_len = 0
    for indx, _source in enumerate(source):
        new_indxs = list(range(len(src_coll),
                               len(src_coll) + len(_source)))
        max_len = max([len(s) for s in _source] + [max_len])
        group_src_indxs.append(torch.LongTensor(new_indxs))
        src_coll += _source

    # adding a fully masked review at the end
    src_coll.append(torch.empty(max_len).fill_(mask_idx))

    return group_src_indxs, src_coll


def collate(samples, pad_idx, eos_idx, mask_idx):
    """Creates a batch specific to the multi-document setting of
    encoding-decoding."""
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return collate_tokens([s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning)

    id = torch.LongTensor([s['id'] for s in samples])

    # SOURCE SEQUENCES #

    # flattened version of the source
    _group_src_indxs, _src = flatten_src([s['source'] for s in samples], mask_idx)
    _group_src_indxs = collate_tokens(_group_src_indxs, pad_idx=len(_src)-1,
                                      left_pad=False, move_eos_to_beginning=False)
    _src_tokens = collate_tokens(_src, pad_idx, eos_idx,
                                 left_pad=False, move_eos_to_beginning=False)
    _src_lengths = torch.LongTensor([s.numel() for s in _src])

    # concatenated version of the source
    src = [torch.cat(s['source']) for s in samples]
    src_tokens = collate_tokens(src, pad_idx, eos_idx,
                                left_pad=False, move_eos_to_beginning=False)
    src_lengths = torch.LongTensor([s.numel() for s in src])

    # TARGET SEQUENCES #

    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=False)
        prev_output_tokens = merge('target', left_pad=False,
                                   move_eos_to_beginning=True)
        tgt_lengths = numel(samples, 'target')
    else:
        target = None
        tgt_lengths = None
        prev_output_tokens = None

    # note that `ntokens` is used for logging of the loss and other metrics in
    # the criterion, so I need to use the number of target tokens
    if tgt_lengths is not None:
        ntokens = int(sum(tgt_lengths))
    else:
        # inference
        ntokens = int(sum(src_lengths))

    batch = {
        'id': id,
        'ntokens': ntokens,
        'nsentences': len(samples),
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            '_group_src_indxs': _group_src_indxs,
            '_src_tokens': _src_tokens,
            '_src_lengths': _src_lengths,
            'prev_output_tokens': prev_output_tokens,
        },
        'target': target,
        'tgt_lengths': tgt_lengths,
    }

    return batch


class AbsDataset(FairseqDataset):
    """Abstractive summarization dataset where input (multiple) documents need
    to be encoded separately.
    """

    def __init__(self, src, tgt, pad_indx, bos_indx, eos_indx, mask_indx,
                 shuffle=False):
        """In inference tgt can be ``None``."""
        super(AbsDataset, self).__init__()

        if tgt is not None:
            assert len(src) == len(tgt)

        self.src = src
        self.tgt = tgt

        self.pad_indx = pad_indx
        self.bos_indx = bos_indx
        self.eos_indx = eos_indx
        self.mask_indx = mask_indx

        self.shuffle = shuffle

    def __getitem__(self, index):
        src_docs = self.src[index]
        assert isinstance(src_docs, list)

        if self.tgt is not None:
            tgt = self.tgt[index]
            assert (tgt[1:-1] >= 1).all()
            assert tgt[0] == self.bos_indx
            assert tgt[-1] == self.eos_indx
        else:
            # inference mode
            tgt = None

        return {'id': index, 'source': src_docs, 'target': tgt}

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(samples=samples, pad_idx=self.pad_indx,
                       eos_idx=self.eos_indx, mask_idx=self.mask_indx)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        src_size = self.src.size(index)
        tgt_size = self.tgt.size(index) if self.tgt else 0
        return tgt_size + src_size

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        tgt_size = self.tgt.size(index) if self.tgt else 0
        # src_size = self.src.size(index)
        # TODO: implement the dynamic checking like checking the maximum length
        # TODO: of the source
        return 0, tgt_size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        # TODO: make an efficient call for sizes in VI sampler
        # return indices[np.argsort(self.src.sizes[indices], kind='mergesort')]
        return indices

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt, 'supports_prefetch',
                             False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)

