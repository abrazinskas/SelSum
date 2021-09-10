from shared_lib.utils.helpers.data import annotate_seq_aspects
from shared_lib.utils.helpers.general import flatten


def get_multi_sent_ngrams(n, sents):
    """Calculates n-grams per sentence, returns their union. Respects sentence
    boundaries."""
    assert n > 0
    assert len(sents) > 0
    coll = set()
    for sent in sents:
        ngrams = get_ngrams(n, sent)
        coll.update(ngrams)
    return coll


def get_ngrams(n, text):
    """Calculates n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    return {"f": f1_score, "p": precision, "r": recall}


def aspect_scores(revs, summ, lexicon, bigrams=True):
    """Calculates aspect precision, recall, f1 of the summary with respect
    to reviews.

    Args:
        revs (list): list of review tokens.
        summ (list): summary, e.g., verdict or pros/cons.
        lexicon (set): aspect lexicon.
    """
    assert isinstance(revs, list)
    rev_aspcts = set(flatten([annotate_seq_aspects(r, lexicon=lexicon,
                                                   bigrams=bigrams)[1]
                              for r in revs]))
    _, summ_aspcts = annotate_seq_aspects(summ, lexicon, bigrams=bigrams)

    # lowercasing to avoid unnecessary duplicates
    rev_aspcts = set({s.lower() for s in rev_aspcts})
    summ_aspcts = set({s.lower() for s in summ_aspcts})
    overlap_set = summ_aspcts & rev_aspcts

    if len(summ_aspcts) > 0:
        pr = len(overlap_set) / len(summ_aspcts)
    else:
        pr = 1.
    if len(rev_aspcts) > 0:
        rec = len(overlap_set) / len(rev_aspcts)
    else:
        rec = 1.
    if (pr + rec) > 0:
        f1 = 2 * rec * pr / (pr + rec)
    else:
        f1 = 0

    return rec, pr, f1, rev_aspcts, summ_aspcts


def aspect_density(asps, seq):
    """Calculates uni-gram and bi-gram densities. Case-insensitive.

    Args:
        asps (set): set of uni-gram and bi-gram aspects.
        seq (list): sequence of tokens.

    Returns:
        unigram: uni-gram density.
        bigram: bi-gram density.
    """
    assert isinstance(asps, set)
    assert isinstance(seq, list)

    seq = [s.lower() for s in seq]

    uni_asps = {a.lower() for a in asps if len(a.split()) == 1}
    bi_asps = {a.lower() for a in asps if len(a.split()) == 2}

    uni_seq = get_ngrams(1, seq)
    bi_seq = get_ngrams(2, seq)

    if len(uni_seq) > 0:
        uni_dens = len(uni_asps) / len(uni_seq)
    else:
        uni_dens = 0.
    if len(bi_seq) > 0:
        bi_dens = len(bi_asps) / len(bi_seq)
    else:
        bi_dens = 0.

    return uni_dens, bi_dens


def compute_pov_distr(tokens):
    """Computes a distribution over 3 points-of-view and one extra slot (other).

    Computation is based on pronouns, and the last class is assigned 100% of
    mass if no pronouns are present.

    Args:
        tokens (list): list of text tokens.

    Returns:
        distr (list): 4 class distribution.
    """
    POVS = [
        {"I", "me", "myself", "my", "mine", "we", "us", "ourselves", "our",
         "ours"},
        {"you", "yourself", "your", "yours"},
        {"he", "she", "it", "him", "her", "his", "hers", "its", "they", "them",
         "their", "theirs"}
    ]
    counts = [0, 0, 0, 0]
    for tok in tokens:
        for indx, pov in enumerate(POVS):
            if tok in pov:
                counts[indx] += 1

    # assigning to the last slot if no POV pronouns were found
    if sum(counts) == 0:
        counts[-1] = 1.

    # normalizing the distribution
    norm = sum(counts)
    distr = [c / float(norm) for c in counts]

    return distr
