import numpy as np


def print_subseq_text_stats(seqs, sep="</s>", tokenizer=None, logger=None):
    """Calculates statistics for generated/gold sub-sequences separated by
    `seq_sep`. Prints to the console.

    Args:
        seqs (list): string sequences.
        sep (str): string separator of sub-sequences.
        logger: if passed, will log instead of printing

    Returns:
        stats (dict): statistic dictionary.
    """
    if logger is not None:
        print_func = logger.info
    else:
        print_func = print

    if tokenizer is None:
        tokenizer = lambda x: x.split()

    seq_len_coll = []
    subseq_len_coll = []
    count_coll = []

    for seq in seqs:
        subseqs = [s.strip() for s in seq.split(sep)]
        subseqs_lens = [len(tokenizer(s)) for s in subseqs]

        subseq_len_coll += subseqs_lens
        seq_len_coll.append(sum(subseqs_lens))
        count_coll.append(len(subseqs))

    # printing statistics
    for coll, title in zip([seq_len_coll, subseq_len_coll, count_coll],
                           ['Seq Length', 'Subseq Length', 'Subseq Count']):
        print_func(f'=== {title} ====')
        print_func(f" | mean: {np.mean(coll):.2f} | std: {np.std(coll):.2f} "
                   f"| min: {min(coll)} | max: {max(coll)} | count: {len(coll)}")
