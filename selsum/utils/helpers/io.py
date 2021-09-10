from shared_lib.utils.constants.features import FEATURE_ORDER
import numpy as np
import os
import torch as T


def read_tags(input_file_path, sep=" "):
    """Reads integer tags. Used by the approximator and extractive summarizer."""
    coll = []
    with open(input_file_path, encoding='utf-8', mode='r') as f:
        for indx, l in enumerate(f, 1):
            try:
                coll.append([int(t) for t in l.strip().split(sep)])
            except Exception as e:
                print(f"error in tag line: {indx}")
    return coll


def make_bin_path(base_path, middle=None):
    """Creates a path to the data binaries."""
    if base_path[-1] == "/":
        base_path = base_path[:-1]
    if middle is None:
        base_path = os.path.join(f'{base_path}-bin')
    else:
        base_path = os.path.join(f'{base_path}-bin', middle)
    return base_path


def get_bin_path(data_path, epoch):
    """Returns the bin data path for a particular split. If no splits, will bin
    the ordinary path.
    """
    if os.pathsep in data_path:
        data_paths = data_path.split(os.pathsep)
        # loading the last epoch data
        if len(data_paths) < epoch:
            data_path = data_paths[-1]
        else:
            data_path = data_paths[epoch - 1]
        bin_path = data_path.split("/")
        bin_path = os.path.join(*bin_path[:-2], f'{bin_path[-2]}-bin',
                                bin_path[-1])
    else:
        bin_path = make_bin_path(base_path=data_path)
    return bin_path
