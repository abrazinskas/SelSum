from fairseq.data import BaseWrapperDataset
from selsum.utils.helpers.collators import collate_features


def collate_extra(samples, batch=None):
    """Adds source and target that are later used during subsampling."""
    batch = batch if batch is not None else {}
    if len(samples) == 0:
        return batch

    # I don't pad neither source nor target because later on I will need
    # to perform subsampling by selecting a subset of source documents
    src = [s['source'] for s in samples]
    id = [s['id'] for s in samples]

    if samples[0]['target'] is not None:
        tgt = [s['target'] for s in samples]
    else:
        tgt = None

    batch['extra'] = {'source': src, 'target': tgt, 'id': id}

    return batch


class InferDataset(BaseWrapperDataset):
    """This dataset is used for running the inference network to obtain
    probabilities of reviews being selected.

    Maximum document length filtering is also optionally performed to remove
    documents along with their features that are far too long.
    """

    def __init__(self, src_ds, feat_ds, tgt_ds=None, max_doc_len=None,
                 debug=False):
        super().__init__(src_ds)
        self.feat_ds = feat_ds
        self.tgt_ds = tgt_ds
        self.max_doc_len = max_doc_len
        self.debug = debug

    def __getitem__(self, index):
        src = self.dataset[index]
        feats = self.feat_ds[index]

        assert len(src) == len(feats)

        # filtering by length
        if self.max_doc_len is not None:
            src_len = [len(_s) for _s in src]
            src = [s for s, l in zip(src, src_len)
                   if l <= self.max_doc_len]
            feats = [f for f, l in zip(feats, src_len)
                     if l <= self.max_doc_len]

        if self.tgt_ds is not None:
            tgt = self.tgt_ds[index]
        else:
            tgt = None

        if self.debug and tgt is not None:
            # adding the target instance to the source
            feats = [[1.] * len(feats[0])] + feats
            src = [tgt] + src

        return {'id': index, 'source': src, 'target': tgt, 'feats': feats}

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch of data
        """
        batch = collate_features(samples)
        batch = collate_extra(samples, batch=batch)
        return batch

    def size(self, index):
        """Assumed that the source sequences will be all documents concatenated
        together, and that no filtering is needed.
        """
        return 0
