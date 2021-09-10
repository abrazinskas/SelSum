from fairseq.data import BaseWrapperDataset
from shared_lib.utils.helpers.data import sent_splitter_multi
import torch as T
import logging

logger = logging.getLogger(__name__)


class SeqSplitter(BaseWrapperDataset):
    """Splits sequences that are concatenated by a special symbol."""

    def __init__(self, dataset, sep_indxs):
        """
        Args:
            dataset:
            sep_indxs (list): subwords of the separator symbol.
        """
        super().__init__(dataset)
        assert isinstance(sep_indxs, list)
        self.sep_indxs = sep_indxs

    def __getitem__(self, index):
        docs = sent_splitter_multi(self.dataset[index], self.sep_indxs,
                                   incl_split_symb=False)
        return docs

    @property
    def sizes(self):
        """
        TODO: implement dynamic computation of the sequence size
        """
        return self.dataset.sizes

    def num_tokens(self, index):
        # TODO: implement dynamic computation of the number of tokens
        return len(self.dataset[index])
