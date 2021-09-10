from fairseq.data import BaseWrapperDataset
import numpy as np


class SeqDataset(BaseWrapperDataset):
    """A simple sequence dataset."""

    def __init__(self, dataset, sizes):
        super().__init__(dataset)
        self.dataset_sizes = sizes

    def __getitem__(self, index):
        return self.dataset[index]

    @property
    def sizes(self):
        return self.dataset_sizes

    def num_tokens(self, index):
        return len(self[index])

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        indices = np.arange(len(self))
        return indices
