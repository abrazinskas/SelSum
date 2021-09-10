from fairseq.data import BaseWrapperDataset
from shared_lib.utils.helpers.data import sent_splitter_multi, concat
import torch as T


class SepReplacer(BaseWrapperDataset):
    """Replaces a sequence of separator symbol indxs with one indx to make
    it easier to perform encoding or decoding.

    The replacement index can be None.
    """

    def __init__(self, dataset, sep_indxs, repl_indx):
        super().__init__(dataset)
        self.sep_indxs = sep_indxs
        self.repl_indx = repl_indx

    def __getitem__(self, index):
        seq = self.dataset[index]
        subseqs = sent_splitter_multi(seq, self.sep_indxs, incl_split_symb=False)
        cat_subseqs = concat(subseqs, self.repl_indx)
        return cat_subseqs

    @property
    def sizes(self):
        return self.dataset.sizes

    def num_tokens(self, index):
        return len(self[index])

    def size(self, index):
        return self.sizes[index]