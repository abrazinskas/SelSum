import re
from shared_lib.utils.helpers.analysis import get_ngrams, cal_rouge
from shared_lib.utils.helpers.topk_ngram_blocker import topk_with_ngram_blocker
from numpy import argsort
import numpy as np


def rouge_greedy_seq_sel(srcs, tgt, ndocs, measure='r', max_ngram=2):
    """Uses ROUGE scores to sequentially and greedily select documents from
    `src`. Adds documents to the bucket while they maximize ROUGE scores.

    Args:
        srcs (list): string documents.
        tgt (str): target summary.
        ndocs: the maximum number of docs to be selected.
        measure: self-explanatory.
        max_ngram: the maximum n-gram to use.
    """
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    tgt = _rouge_clean(tgt).lower().split()
    srcs = [_rouge_clean(s).lower().split() for s in srcs]

    src_ngrams = {n: [get_ngrams(n, _src) for _src in srcs]
                 for n in range(1, max_ngram + 1)}
    tgt_ngrams = {n: get_ngrams(n, tgt) for n in range(1, max_ngram + 1)}

    selected = []
    max_rouge = 0.0
    for s in range(ndocs):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(srcs)):
            if i in selected:
                continue
            c = selected + [i]

            rouge_score = 0
            for n in range(1, max_ngram + 1):
                candidates_ngrams = [src_ngrams[n][idx] for idx in c]
                candidates_ngrams = set.union(*map(set, candidates_ngrams))
                score = cal_rouge(candidates_ngrams, tgt_ngrams[n])[measure]
                rouge_score += score

            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return sorted(selected)
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def rouge_top_k(srcs, tgt, ndocs, measure='f', ngram_block=None, ngram=1):
    """Returns `ndocs` src documents that have the highest ROUGE score w.r.t the
    target summary. Optionally, applies n-gram blocking.

    Args:
        srcs (list): source documents.
        tgt (str): the target summary.
        ndocs (int): that many documents to select.
        measure: r, p, f. For instance, if recall is selected, it will result
            in reviews that are longer and provide a better coverage of summary
            n-grams.
        ngram_block:
        ngram: what n-gram base the selection on
    Returns:
        sel_indxs (list): list of selected
    """
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    tgt = _rouge_clean(tgt).lower().split()
    srcs_toks = [_rouge_clean(s).lower().split() for s in srcs]

    # computing n-grams
    src_ngrams = [get_ngrams(ngram, _src) for _src in srcs_toks]
    tgt_ngrams = get_ngrams(ngram, tgt)

    scores = [cal_rouge(_ngrams, tgt_ngrams)[measure] for _ngrams in src_ngrams]

    if ngram_block is not None and len(srcs_toks) > ndocs:
        sel_indxs = topk_with_ngram_blocker(scores, srcs_toks, k=ndocs,
                                            ngram_block=ngram_block)
    else:
        sel_indxs = argsort(scores)[-ndocs:]

    return sorted(sel_indxs)


def mixture_sample(srcs, tgt, ndocs, oracle_mix_prob, measure='r', ngram=1):
    """Returns a sample without replacement of size `ndocs` from the mixture
    distribution:

       p(s) = p(s|oracle) p(oracle) + p(s|random) (1 - p(oracle))

    At each step, the previouly selected src documents are blocked from being
    selected.

    Args:
        srcs:
        tgt:
        oracle_mix_prob:
        measure: oracle measure 'r', 'p', 'f'.
        ngram: ROUGE n-gram
    """
    assert len(srcs) > ndocs
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    tgt = _rouge_clean(tgt).lower().split()
    srcs_toks = [_rouge_clean(s).lower().split() for s in srcs]

    # computing n-grams
    src_ngrams = [get_ngrams(ngram, _src) for _src in srcs_toks]
    tgt_ngrams = get_ngrams(ngram, tgt)

    # calculating the oracle distribution
    oracle_scores = np.array([cal_rouge(_ngrams, tgt_ngrams)[measure]
                              for _ngrams in src_ngrams])
    uni_scores = np.ones(len(oracle_scores))
    mask = np.zeros(len(oracle_scores))

    sampled_indxs = []
    for _ in range(ndocs):

        # creating distributions
        if oracle_mix_prob > 0 and np.random.binomial(n=1, p=oracle_mix_prob):
            # oracle distribution
            _sample_distr = oracle_scores * (1 - mask)
            _denom = _sample_distr.sum(-1, keepdims=True)
            if _denom > 0:
                _sample_distr = _sample_distr / _denom
            else:
                # flat categorical
                _sample_distr = uni_scores * (1 - mask)
                _sample_distr = _sample_distr / (1 - mask).sum(-1, keepdims=True)
        else:
            # flat categorical
            _sample_distr = uni_scores * (1 - mask)
            _sample_distr = _sample_distr / (1 - mask).sum(-1, keepdims=True)

        sel_indx = np.random.choice(range(len(_sample_distr)), p=_sample_distr)

        mask[sel_indx] = 1
        
        sampled_indxs.append(sel_indx)

    return sampled_indxs
