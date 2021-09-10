import operator
from collections import OrderedDict
from copy import deepcopy


def is_in_range(x, min=None, max=None):
    """Returns ``True`` if `x` is in range; ``False`` otherwise."""
    if min and x < min or max and x > max:
        return False
    return True


def flatten(my_list):
    """Flattens nested lists."""
    curr_items = []
    for x in my_list:
        if isinstance(x, list) and not isinstance(x, (str, bytes)):
            curr_items += flatten(x)
        else:
            curr_items.append(x)
    return curr_items


def sort_hash(hash, by_key=True, reverse=False):
    if by_key:
        indx = 0
    else:
        indx = 1
    return sorted(hash.items(), key=operator.itemgetter(indx), reverse=reverse)


def listify(val):
    """If val is an element the func wraps it into a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    return [val]


def merge_dicts(dct1, dct2):
    """Merges two dictionaries assuming no overlap between keys."""
    if isinstance(dct1, OrderedDict) and isinstance(dct2, OrderedDict):
        new_dict = OrderedDict()
    else:
        new_dict = {}
    for dct in [dct1, dct2]:
        for k,v in dct.items():
            assert k not in new_dict
            new_dict[k] = deepcopy(dct[k])
    return new_dict