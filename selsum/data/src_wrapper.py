from fairseq.data import BaseWrapperDataset
import torch as T


class SrcWrapper(BaseWrapperDataset):
    """Wraps a list of sequences with `bos_indx` and `eos_indx`."""

    def __init__(self, dataset, bos_indx, eos_indx):
        super(SrcWrapper, self).__init__(dataset)
        self.bos_indx = bos_indx
        self.eos_indx = eos_indx

    def __getitem__(self, index):
        src_docs = self.dataset[index]
        assert isinstance(src_docs, list)
        src_docs = [self.wrap_start_end(doc) for doc in src_docs]
        return src_docs

    def wrap_start_end(self, tens):
        has_start = tens[0] == self.bos_indx
        has_end = tens[-1] == self.eos_indx
        if has_start or has_end:
            raise ValueError("An entry has either start or end symbol.")
        res = T.cat((tens.new([self.bos_indx]), tens, tens.new([self.eos_indx])))
        return res
