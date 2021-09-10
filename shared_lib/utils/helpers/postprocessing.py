import re


def split_asp_seq(seq):
    """Splits aspect tagged sequence by spaces except when in brackets."""
    words = re.split(r'\s+(?=[^\[\]]*(?:\[|$))', seq)
    return words


def mark_asp_matches(seq, lex):
    """Marks matching aspects in the annotated sequence."""
    words = split_asp_seq(seq)
    new_seq = []
    for word in words:
        if re.match('^\[[\w\s]+\]$', word):
            if word[1:-1].lower() in lex:
                word = f'{word}(+)'
            else:
                word = f'{word}(-)'
        new_seq.append(word)
    new_seq = " ".join(new_seq)
    return new_seq
