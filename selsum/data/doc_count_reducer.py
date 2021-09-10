import numpy as np
from fairseq.data import BaseWrapperDataset
import torch as T
import logging

logger = logging.getLogger(__name__)


class DocCountReducer(BaseWrapperDataset):
    """Reduces the number of documents in instances."""

    def __init__(self, dataset, truncation_length):
        super().__init__(dataset)
        assert truncation_length is not None
        self.truncation_length = truncation_length
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        if not isinstance(item, (list, np.ndarray, T.Tensor)):
            raise TypeError("Items should be an iterable collector.")
        item_len = len(item)
        if item_len > self.truncation_length:
            item = item[:self.truncation_length]
        return item

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.truncation_length)

    def __len__(self):
        return len(self.dataset)