from shared_lib.utils.helpers.data import annotate_seq_aspects


def aspect_score(hyp, ref, lexicon, default_score=1.):
    """Scores (hyp, ref) tuple with aspect recall, precision, and f1.

    Args:
        hyp (list): tokens of the hypothesis sequence.
        ref (list): tokens of the reference sequence.
        lexicon (set): aspect lexicon.
        default_score (float): the default value used when no aspects are in
            the reference or hypothesis summary.
    """
    hyp_aspcts = set(annotate_seq_aspects(hyp, lexicon=lexicon, bigrams=True)[1])
    ref_aspcts = set(annotate_seq_aspects(ref, lexicon=lexicon, bigrams=True)[1])

    overlap_set = ref_aspcts & hyp_aspcts

    if len(hyp_aspcts) > 0:
        pr = len(overlap_set) / len(hyp_aspcts)
    else:
        pr = default_score
    if len(ref_aspcts) > 0:
        rec = len(overlap_set) / len(ref_aspcts)
    else:
        rec = default_score
    if (pr + rec) > 0:
        f1 = 2 * rec * pr / (pr + rec)
    else:
        f1 = 0

    return {'r': rec, 'p': pr, 'f': f1}
