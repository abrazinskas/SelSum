from shared_lib.utils.constants.features import FEATURE_ORDER


def format_feats(scores):
    """Formats features to string such that they could be stored in files."""
    coll = []
    for s in scores:
        coll.append(" ".join([f'{s[k]:.5f}' for k in FEATURE_ORDER]))
    coll_str = '\t'.join(coll)
    return coll_str
