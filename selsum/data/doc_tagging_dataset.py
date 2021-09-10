from fairseq.data import data_utils, FairseqDataset
from selsum.data.abs_dataset import flatten_src
from fairseq.data.data_utils import collate_tokens
from selsum.utils.helpers.data import filter_long_docs
import torch as T
import numpy as np
import logging

logger = logging.getLogger(__name__)


def collate(samples, pad_indx, eos_indx, mask_indx):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_idx=pad_indx):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_indx, left_pad, move_eos_to_beginning,
        )

    id = T.LongTensor([s['id'] for s in samples])

    # SOURCE SEQUENCES #

    # flattened version of the source
    # an extra review (the last one) is added for padding
    group_src_indxs, _src = flatten_src([s['source'] for s in samples], mask_indx)
    group_src_indxs = collate_tokens(group_src_indxs, pad_idx=-1,
                                     left_pad=False, move_eos_to_beginning=False)
    src_tokens = collate_tokens(_src, pad_idx=pad_indx, eos_idx=eos_indx,
                                left_pad=False, move_eos_to_beginning=False)
    src_lengths = T.LongTensor([s.numel() for s in _src])

    # TARGET SEQUENCES #

    # creating the actual target (binary tags)
    if samples[0].get('target', None) is not None:
        tgt = merge('target', left_pad=False, pad_idx=-1)
    else:
        tgt = None

    ntokens = (group_src_indxs != -1).sum()

    batch = {
        'id': id,
        'ntokens': ntokens,
        'nsentences': len(samples),
        'net_input': {
            'group_src_indxs': group_src_indxs,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': tgt,
    }

    return batch


FILTER_OUT_SIZE = 10000


class DocTaggingDataset(FairseqDataset):
    """Dataset for the document tagging task. Each document has an associated
    tag with it.

    In this dataset, tokens correspond to the number of documents instead of the
    number of subwords to be used in each batch.

    Also,
    """

    def __init__(self, src, tgt, pad_indx, eos_indx, mask_indx,
                 shuffle=False, min_docs=None, max_doc_len=None):
        """
        Args:
            min_docs: if an instance has less documents than `min_docs`, it will
                be filtered out. Please note that the warning displayed will be misleading:
                    1217 samples have invalid sizes and will be skipped, max_positions=200
            max_doc_len (int): if passed, will remove documents that are very long
                together with their tags
        """
        super().__init__()
        self.src = src
        self.tgt = tgt
        self.pad_indx = pad_indx
        self.eos_indx = eos_indx
        self.mask_indx = mask_indx
        self.shuffle = shuffle
        self.min_docs = min_docs
        self.max_doc_len = max_doc_len

    def __getitem__(self, index):
        src = self.src[index]
        assert isinstance(src, list)
        if self.tgt is not None:
            tgt = self.tgt[index]
            assert len(tgt) == len(src)
            if self.max_doc_len is not None:
                src, tgt = filter_long_docs(src, tgt, max_doc_len=self.max_doc_len)
                tgt = T.tensor(tgt)
        else:
            tgt = None
            if self.max_doc_len is not None:
                src = [s for s in src if len(s) <= self.max_doc_len]

        return {'id': index, 'source': src, 'target': tgt}

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(samples=samples, pad_indx=self.pad_indx,
                       eos_indx=self.eos_indx, mask_indx=self.mask_indx)

    def __len__(self):
        return len(self.src)

    def size(self, index):
        # TODO: make it compute things dynamically
        """Returns the number of documents based on target tags."""
        size = self.tgt.size(index)
        if self.min_docs is not None and size < self.min_docs:
            return FILTER_OUT_SIZE
        return size

    def num_tokens(self, index):
        # TODO: make it compute things dynamically
        """"""
        size = self.tgt.size(index)
        return size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.tgt.sizes[indices], kind='mergesort')]

