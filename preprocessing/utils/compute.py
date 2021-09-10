from shared_lib.utils.constants.general import SEQ_SEP
from shared_lib.metrics.google_rouge_comps.rouge_scorer import RougeScorer
from shared_lib.utils.helpers.analysis import aspect_density, annotate_seq_aspects
from shared_lib.utils.constants.features \
    import (SRC_AD1, VERD_AD1, PC_AD1, SRC_VERD_LEN_DIFF, SRC_PC_LEN_DIFF,
            SRC_VERD_AR, SRC_VERD_AP, SRC_VERD_R1_R, SRC_VERD_R1_P, SRC_VERD_R2_R, SRC_VERD_R2_P,
            SRC_PC_AR, SRC_PC_AP, SRC_PC_R1_R, SRC_PC_R1_P, SRC_PC_R2_R, SRC_PC_R2_P,
            REST_AP, REST_AR, REST_R1_P, REST_R1_R, REST_R2_P, REST_R2_R)
from shared_lib.utils.helpers.general import flatten
from preprocessing.utils.score_funcs import aspect_score
import numpy as np


def get_max_lens(src, tgt):
    """Computes the maximum lengths for source documents, verdicts, and
    pros & cons. For instance, based on the training and validation entries.

    Args:
        src (list): source documents, each entry separated by `SEQ_SEP`.
        tgt (list): target summaries, verdicts, pros and cons separated by
            `SEQ_SEP`.
    """
    src_lens = flatten([[len(_s.strip().split()) for _s in _src.split(SEQ_SEP)]
                        for _src in src])
    src_max_len = max(src_lens)

    tgt_lens = np.array([[len(_t.strip().split()) for _t in _tgt.split(SEQ_SEP)]
                         for _tgt in tgt])
    verd_max_len = max(tgt_lens[:, 0])
    pc_max_len = tgt_lens[:, 1:].sum(-1).max()

    return src_max_len, verd_max_len, pc_max_len


def get_features(src, tgt, src_max_len, verd_max_len, pc_max_len, lex):
    """Computes features for each source document; separately paired with
    verdicts, pros and cons (target).

    There are three types of features:
        1) source vs target
        2) source and target alone
        3) source vs other source documents (global features)

    Args:
        src (str): source string with concatenated documents.
        tgt (str): concatenated verdict, pros & cons.
        src_max_len (int): maximum length of the source document.
        verd_max_len (int): maximum length of the verdict summary.
        pc_max_len (int): maximum length of the pros&cons summary.
        lex (set): fine-grained lexicon terms.

    Returns:
        coll (list): source document features (dicts).
    """
    default_aspect_score = -1  # to indicate no aspects in reference/hypothesis
    rouge_scorer = RougeScorer(['rouge1', 'rouge2'])

    verd, pros, cons = [t.strip() for t in tgt.lower().split(SEQ_SEP)]
    pros_cons = f"{pros} {cons}"

    verd_toks = verd.split()
    pc_toks = pros_cons.split()

    coll = list()

    src_docs = [s.lower().strip() for s in src.split(SEQ_SEP)]
    src_doc_toks = [s.split() for s in src_docs]

    for indx, (s_doc, s_toks) in enumerate(zip(src_docs, src_doc_toks)):
        feats = {}

        # ------------ #
        #  src vs tgt  #
        # ------------ #

        # aspect-based scores
        verd_ascore = aspect_score(hyp=s_toks, ref=verd_toks, lexicon=lex,
                                   default_score=default_aspect_score)
        pc_ascore = aspect_score(hyp=s_toks, ref=pc_toks, lexicon=lex,
                                 default_score=default_aspect_score)

        feats[SRC_VERD_AP] = verd_ascore['p']
        feats[SRC_VERD_AR] = verd_ascore['r']
        feats[SRC_PC_AP] = pc_ascore['p']
        feats[SRC_PC_AR] = pc_ascore['r']

        # rouge-based scores
        verd_rouge = rouge_scorer.score(target=verd, prediction=s_doc)
        verd_r1 = verd_rouge['rouge1']
        verd_r2 = verd_rouge['rouge2']

        feats[SRC_VERD_R1_R] = verd_r1.recall
        feats[SRC_VERD_R1_P] = verd_r1.precision
        feats[SRC_VERD_R2_R] = verd_r2.recall
        feats[SRC_VERD_R2_P] = verd_r2.precision

        pc_rouge = rouge_scorer.score(target=pros_cons, prediction=s_doc)
        pc_r1 = pc_rouge['rouge1']
        pc_r2 = pc_rouge['rouge2']

        feats[SRC_PC_R1_R] = pc_r1.recall
        feats[SRC_PC_R1_P] = pc_r1.precision
        feats[SRC_PC_R2_R] = pc_r2.recall
        feats[SRC_PC_R2_P] = pc_r2.precision

        # -------------------- #
        #   src and tgt alone  #
        # -------------------- #

        _, s_asps = annotate_seq_aspects(s_toks, lexicon=lex, bigrams=False)
        _, verd_asps = annotate_seq_aspects(verd_toks, lexicon=lex,
                                            bigrams=False)
        _, pc_asps = annotate_seq_aspects(pc_toks, lexicon=lex, bigrams=False)

        s_uni_dens, _ = aspect_density(set(s_asps), s_toks)
        verd_uni_dens, _ = aspect_density(set(verd_asps), verd_toks)
        pc_uni_dens, _ = aspect_density(set(pc_asps), pc_toks)

        s_verd_len_diff = len(s_toks) / src_max_len - len(
            verd_toks) / verd_max_len
        s_pc_len_diff = len(s_toks) / src_max_len - len(pc_toks) / pc_max_len

        assert abs(s_verd_len_diff) <= 1.
        assert abs(s_pc_len_diff) <= 1.

        feats[SRC_VERD_LEN_DIFF] = s_verd_len_diff
        feats[SRC_AD1] = s_uni_dens
        feats[VERD_AD1] = verd_uni_dens

        feats[SRC_PC_LEN_DIFF] = s_pc_len_diff
        feats[PC_AD1] = pc_uni_dens

        # ----------------- #
        #   src vs others   #
        # ----------------- #

        other_src_docs_str = " ".join([_s for i, _s in enumerate(src_docs)
                                       if i != indx])
        other_src_doc_toks = flatten([_s_toks for i, _s_toks
                                      in enumerate(src_doc_toks) if i != indx])

        # aspect-based scores
        rest_ascore = aspect_score(hyp=s_toks, ref=other_src_doc_toks,
                                   lexicon=lex,
                                   default_score=default_aspect_score)
        feats[REST_AP] = rest_ascore['p']
        feats[REST_AR] = rest_ascore['r']

        # rouge-based scores
        rest_rouge = rouge_scorer.score(target=other_src_docs_str,
                                        prediction=s_doc)

        rest_r1 = rest_rouge['rouge1']
        feats[REST_R1_R] = rest_r1.recall
        feats[REST_R1_P] = rest_r1.precision

        rest_r2 = rest_rouge['rouge2']
        feats[REST_R2_R] = rest_r2.recall
        feats[REST_R2_P] = verd_r2.precision

        assert all([abs(v) <= 1 for v in feats.values()])

        coll.append(feats)

    return coll
