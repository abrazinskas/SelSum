def extract_summs_from_src(src, tags, empty_placeholder="."):
    """Returns 3 summaries based on passed tags. By convention, verdicts, pros,
    and cons are tagged as 1, 2, 3, respectively.

    Args:
        src (list): src documents
        tags (list): integer tags
        empty_placeholder: symbol to use if a summary is missing
    """
    assert len(src) == len(tags)
    verd = []
    pros = []
    cons = []
    for indx, t in enumerate(tags):
        if t == 1:
            verd.append(src[indx])
        if t == 2:
            pros.append(src[indx])
        if t == 3:
            cons.append(src[indx])
    verd = " ".join(verd) if verd else empty_placeholder
    pros = " ".join(pros) if pros else empty_placeholder
    cons = " ".join(cons) if cons else empty_placeholder
    return verd, pros, cons


def format_all_summs(summs, sep_symb="</s>", empty_symb=" "):
    """Formats a joint summary string and stores to three collectors.

    Args:
        summs: list of summary strings.
        sep_symb: separator to be used for pros & cons.
        empty_symb: what symbol to use if the string is empty.

    Returns:
        verds: verdicts.
        pros_cons: pros and cons where the latter follows the former.
        pros_cons_cat: pros and cons concatenated together.
    """
    verd_coll = []
    pros_coll = []
    cons_coll = []

    count_stats = []

    for summ in summs:
        verd = empty_symb
        pros = empty_symb
        cons = empty_symb

        _summs = [s.strip() for s in summ.split(sep_symb)]

        count_stats.append(len(_summs))

        if len(summs) >= 1:
            verd = _summs[0]
        if len(summs) >= 2:
            pros = _summs[1]
        # this is a case where more than 3 sub-sequences are in the output
        if len(summs) >= 3:
            cons = " ".join(_summs[2:])

        verd = verd.strip()
        pros = pros.strip()
        cons = cons.strip()

        if not len(verd):
            verd = empty_symb
        if not len(pros):
            pros = empty_symb
        if not len(cons):
            cons = empty_symb

        verd_coll.append(verd)
        pros_coll.append(pros)
        cons_coll.append(cons)

    return verd_coll, pros_coll, cons_coll
