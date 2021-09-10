# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from fairseq.data import BaseWrapperDataset
from shared_lib.utils.helpers.data import reduce_docs
import torch as T
import logging

logger = logging.getLogger(__name__)


class DocReducer(BaseWrapperDataset):
    """Reduces the number of review documents to fit a budget. Optionally sorts
    and retains the longest ones. Assumes that reviews are split already.

    Also removed too long documents.
    """

    def __init__(self, dataset, max_budgets=None, max_doc_len=None,
                 sort_docs=False):
        """
        Args:
            dataset (Dataset):
            max_doc_len (int): if set, will remove all documents that are too long
            max_budgets (list): maximum allowed length in subwords that.
                includes the sep symbol ids. Individual per each data-point.
        """
        super().__init__(dataset)
        assert max_budgets is None or isinstance(max_budgets, list)
        self.max_budget = max_budgets
        self.max_doc_len = max_doc_len
        self.sort_docs = sort_docs

    def __getitem__(self, index):
        docs = self.dataset[index]
        assert isinstance(docs, list)

        if self.max_doc_len is not None:
            docs = [d for d in docs if len(d) <= self.max_doc_len]

        if self.max_budget is not None:
            budget = self.max_budget[index] \
                if isinstance(self.max_budget, list) else self.max_budget
            total_len = sum([len(d) for d in docs])
            if total_len > budget:
                docs = reduce_docs(docs, max_budget=budget, offset=0,
                                   sort_docs=self.sort_docs)
        return docs

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.max_budget)

    def __len__(self):
        return len(self.dataset)



