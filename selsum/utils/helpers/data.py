import torch as T
from nltk import sent_tokenize


def sent_split_docs(doc_str, seq_sep, subseq_sep):
    """Splits document string by sents and joins with proper separator."""
    coll = []
    for doc in doc_str.split(seq_sep):
        doc = doc.strip()
        tmp = []
        for sent in sent_tokenize(doc):
            tmp.append(sent)
        coll.append(f' {subseq_sep.strip()} '.join(tmp))
    coll = f' {seq_sep.strip()} '.join(coll)
    return coll


def numel(samples, key):
    """Counts the total number of elements in each element selected by `key`.

    Args:
        samples: dict with lists.
        key: what values to convert.

    Returns:
        LongTensor with counts.
    """
    return T.LongTensor([s[key].numel() for s in samples])


def chunk_data(data, size):
    """Creates a list of chunks of the specified `size`.

    Args:
        data (list): self-explanatory.
        size (int): desired size of each chunk, the last one can be <= `size`.

    Returns:
        coll (list): lists of lists.
    """
    coll = []
    start_indx = 0
    while start_indx < len(data):
        slice_range = range(start_indx, min(start_indx + size, len(data)))
        chunk = [data[i] for i in slice_range]
        coll.append(chunk)
        start_indx += size
    return coll


def pad_feats(feats, pad_val=0.):
    """Pads the second dimension with the vector of zeros to eliminate the uneven
    number of reviews per product. Assumes that each feature vector has the
    same dimensionality.

    Args:
        feats (list): list of floats
            (batch_size, arbitrary rev_num*, feat_dim)
        pad_val (float): the value that should be used for padding.

    Returns:
        feat_coll: (batch_size, rev_num, feat_dim)
        mask_coll: (batch_size, rev_num

    """
    pad = [pad_val] * len(feats[0][0])

    max_l = max([len(f) for f in feats])

    feat_coll = []
    mask_coll = []

    for _feats in feats:
        _mask = [False] * len(_feats)
        _pad_num = max_l - len(_feats)
        if _pad_num > 0:
            _pad = [pad for _ in range(_pad_num)]
            _new_feats = _feats + _pad
            _mask += [True] * len(_pad)
        else:
            _new_feats = _feats
        feat_coll.append(_new_feats)
        mask_coll.append(_mask)

    feat_coll = T.FloatTensor(feat_coll)
    mask_coll = T.BoolTensor(mask_coll)

    return feat_coll, mask_coll


def filter_long_docs(src, tgt, max_doc_len):
    """Filters too long documents together with their targets. Returns lists."""
    new_src = []
    new_tgt = []
    for s, t in zip(src, tgt):
        if len(s) > max_doc_len:
            continue
        new_src.append(s)
        new_tgt.append(t)
    return new_src, new_tgt
