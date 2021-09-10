from shared_lib.utils.helpers.analysis import get_ngrams
from numpy import argsort


def _topk_with_ngram_blocker(sorted_sents, k, ngram_block):
    """Selects top-k sentences, blocks n-gram repeated ones.
    
    Args:
        sorted_sents (list/array): lists of ids/tokens. Assumes that they are
            sorted by scores already.
        k: that many sentences to select.
        ngram_block: integer indicating what n-grams to block.

    Returns:
        list of selected sentence indices.
    """
    coll_ngrams = set()
    coll = []
    for indx, sent in enumerate(sorted_sents):
        if len(coll) == k:
            break
        curr_ngrams = get_ngrams(ngram_block, sent)
        if len(coll_ngrams.intersection(curr_ngrams)) > 0:
            continue
        coll.append(indx)
        coll_ngrams = curr_ngrams.union(coll_ngrams)
    return coll


def topk_with_ngram_blocker(scores, sents, k, ngram_block):
    """Selects top-k scored sentences by applying the ngram blocker to prevent
        non-unique sentences.

    Args:
        scores (list): float scores, e.g., probabilities.
        sents (list): list of tokens/ids corresponding to each sentence.
        k (int): that many sentences to select.
        ngram_block (int): block a sentence if it has an n-gram already present
            in other selected sentences.

    Returns:
        sel_indxs (list): selected sentence indices.
    """
    assert len(scores) == len(sents)
    if k >= len(sents):
        return range(len(sents))
    indxs = argsort(scores)[::-1]
    sorted_sents = [sents[i] for i in indxs]
    sel_indxs = _topk_with_ngram_blocker(sorted_sents, k=k, ngram_block=ngram_block)
    sel_indxs = [indxs[i] for i in sel_indxs]
    return sel_indxs
