from nltk import word_tokenize
import torch as T


def sent_splitter_multi(doc, split_symb, incl_split_symb=False):
    """Splits `doc` by sentences where boundaries are based on multiple units in
    `split_symb`. For example, if a separator is a sequence of subword ids.

    Args:
        doc (list/array/tensor): tokens/ids.
        split_symb (list): subwords corresponding to one word like </s>.
        incl_split_symb (bool): whether to include split symbols to sentences.

    Returns:
        sents (list): sentences.
    """
    sents = []

    ss_start_indx = 0
    ss_point_indx = 0
    doc_point_indx = 0
    offset = len(split_symb) if not incl_split_symb else 0

    while doc_point_indx < len(doc):
        s = split_symb[ss_point_indx]
        d = doc[doc_point_indx]
        if d == s:
            ss_point_indx += 1
        else:
            ss_point_indx = 0
        doc_point_indx += 1
        if ss_point_indx == len(split_symb):
            sents.append(doc[ss_start_indx: doc_point_indx - offset])
            ss_start_indx = doc_point_indx
            ss_point_indx = 0
    if ss_point_indx != doc_point_indx:
        sents.append(doc[ss_start_indx:])
    return sents


def truncate_subwords(doc_sents, max_subwords, bpe, sep_symb):
    """Truncates subwords of the doc` to fit `max_subwords` budget.
    The budget includes subwords of `sep_symb`. And `doc` is internally mapped
    to subwords.

    Args:
         doc_sents: list of sentence strings (words).
         max_subwords: the allowed budget in subwords.
         bpe: fairseq bpe encoder.
         sep_symb: separation symbol with padding.
    Returns:
        list of (word) sentence strings.
    """
    assert " " in sep_symb
    # mapping the document sentences to BPEs
    doc_str = f"{sep_symb} ".join(doc_sents)
    doc_bpe = bpe.encode(doc_str)
    doc_bpe_symb = bpe.encode(sep_symb)
    doc_bpe_sents = [s.strip() for s in doc_bpe.split(doc_bpe_symb)]

    # selecting sentences that fit the budget
    coll = []
    curr_len = 0
    for indx, sent in enumerate(doc_bpe_sents):
        bpe_sent_len = len(sent.split())
        # adding offset to all but the first sentence
        if indx > 0:
            bpe_sent_len += len(doc_bpe_symb.split())
        if bpe_sent_len <= max_subwords - curr_len:
            coll.append(bpe.decode(sent).strip())
            curr_len += bpe_sent_len
        else:
            break

    # sanity check
    # act_len = len(bpe.encode(f" {sep_symb} ".join(coll)).split())
    # assert act_len == curr_len

    return coll


def reduce_docs(docs, max_budget, offset, sort_docs=False):
    """Reduces the number of `docs` (subwords) to fit maximum budget by
    optionally sorting them by length, and then removing the shortest ones
    until `max_budget` is exceeded. Otherwise, will go left-to-right to preserve
    documents.

    Args:
         docs (list/array): tokenized documents.
         max_budget (int): the allowed budget in words/subwords.
         offset (int): additional offset for each doc.
         sort_docs (bool): if set, will sort documents by length and will
            attempt to remove the short ones.

    Returns:
        list of selected docs (preserved order).
    """
    if sort_docs:
        doc_lens = [len(d) for d in docs]
        order_indxs = sorted(range(len(doc_lens)), key=lambda k: doc_lens[k],
                             reverse=True)
    else:
        order_indxs = range(len(docs))
    curr_len = 0
    indx_coll = []
    for i, indx in enumerate(order_indxs):
        doc = docs[indx]
        if isinstance(doc, str):
            raise ValueError("Documents must be tokenized.")
        doc_len_w_offset = 0
        # adding the separator symbol offset to all but the first doc
        if i > 0:
            doc_len_w_offset += offset
        doc_len_w_offset += len(doc)
        if doc_len_w_offset <= max_budget - curr_len:
            curr_len += doc_len_w_offset
            indx_coll.append(indx)
        else:
            break
    indx_coll = sorted(indx_coll)
    docs = [docs[j] for j in indx_coll]
    return docs, indx_coll


def annotate_seq_aspects(seq, lexicon, bigrams=True):
    """Annotates sequence words using a passed lexicon.

    Assumes that the lexicon contains both unigram and bigram elements.

    Args:
        seq (list): list of words.
        lexicon (set): aspects or opinions.
        bigrams (bool): if set to ``True``, will concatenate bi-grams to
            check if it's present in the lexicon.

    Returns:
        new_seq (str): annotated sequence where matching words are 
            wrapped in [].
        aspects (list): list of aspects found in `seq`.
    """
    assert isinstance(lexicon, set)
    if not isinstance(seq, list):
        raise ValueError("Sequence must be tokenized.")
    
    annotated_seq = []
    aspects = []

    indx = 0
    while indx < len(seq):
        curr_word = seq[indx]
        next_word = seq[indx + 1] if indx + 1 < len(seq) else None

        # checking bi-grams
        if next_word:
            bigram = f'{curr_word} {next_word}'
            if bigram.lower() in lexicon or \
                (bigrams and f'{curr_word}{next_word}' in lexicon):
                annotated_seq.append(f'[{bigram}]')
                aspects.append(bigram)
                indx += 2
                continue

        # checking uni-grams
        if curr_word.lower() in lexicon:
            aspects.append(curr_word)
            curr_word = f'[{curr_word}]'

        annotated_seq.append(curr_word)
        indx += 1

    annotated_seq = " ".join(annotated_seq)
    return annotated_seq, aspects


def concat(lst, cat_symb=None, append_to_end=False):
    """Concatenates `lst` of Tensors, optionally with a join symbol.

    Args:
        lst: list of Tensors to concatenate.
        cat_symb: concatenation symbol.
        append_to_end: if set to ``True``, it will add the `cat_symb` to the end
            of the concatenated sequence.
    Returns:
        cat_tens (Tensor): concatenated tensor where sub-tensors are separated
            by 'cat_symb'.
    """
    assert isinstance(lst, list)
    if cat_symb is not None:
        new_lst = []

        if not isinstance(cat_symb, list):
            cat_symb = [cat_symb]
        cat_symb = T.tensor(cat_symb).to(lst[0].device)

        for indx, e in enumerate(lst):
            new_lst.append(e)
            if indx == len(lst)-1 and not append_to_end:
                continue
            new_lst.append(cat_symb)
    else:
        new_lst = lst
    cat_tens = T.cat(new_lst)
    return cat_tens
