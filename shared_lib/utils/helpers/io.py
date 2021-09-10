import pandas as pd
from csv import QUOTE_NONE
from shared_lib.utils.helpers.paths_and_files import safe_mkfdir
from filelock import FileLock
import logging
import gzip
import json
from shared_lib.utils.helpers.paths_and_files import get_file_paths, get_file_name
from multiprocessing.pool import Pool
import torch as T
from shared_lib.utils.constants.features import FEATURE_ORDER


# disabling logging in the module (too chatty)
logging.getLogger('filelock').setLevel(logging.WARNING)


def read_table_data(input_file_path, nrows=None):
    """Standard reading of table data using pandas."""
    ds = pd.read_csv(input_file_path, sep="\t", encoding='utf-8',
                     quoting=QUOTE_NONE, nrows=nrows, keep_default_na=False)
    return ds


def write_table_data(ds, output_file_path):
    """Writing the dataset to a file."""
    safe_mkfdir(output_file_path)
    ds.to_csv(output_file_path, encoding='utf-8', sep="\t", quoting=QUOTE_NONE,
              index=False)


def read_data(input_file_path, nrows=None, encoding='utf-8', skip_empty=False):
    """Reads lines in a file. Returns a list with string lines."""
    coll = []
    with open(input_file_path, encoding=encoding, mode='r') as f:
        for indx, line in enumerate(f):
            if nrows is not None and indx >= nrows:
                break
            line = line.strip()
            if not line:
                continue
            if skip_empty and not line:
                continue
            coll.append(line)
    return coll


def write_data(data, file_path, encoding='utf-8'):
    """Writes list data to a file."""
    safe_mkfdir(file_path)
    with open(file_path, mode='w', encoding=encoding) as f:
        for du in data:
            f.write(f'{du}\n')


def append_to_file(file_path, line, lock_file_path=None, encoding='utf-8'):
    """Appends a line to a file that has a concurrency lock."""
    safe_mkfdir(file_path)
    if lock_file_path is not None:
        safe_mkfdir(lock_file_path)
        lock = FileLock(lock_file_path)
        with lock:
            open(file_path, 'a', encoding=encoding).write(line + "\n")
    else:
        open(file_path, 'a', encoding=encoding).write(line + "\n")


def iter_gz_file(file_path):
    with gzip.open(file_path, mode='r') as f:
        for line in f:
            yield json.loads(line)


def load_json_file(file_path):
    """Loads a dictionary from a JSON file."""
    name = get_file_name(file_path)
    with open(file_path, encoding='utf-8') as f:
        return name, json.load(f)


def read_json_files(folder_path, uppercase_name=False, processes=10):
    """Reads JSON files asynchronously on multiple processes.

    Args:
        folder_path: folder with JSON files.
        uppercase_name: whether to uppercase file names that are returned as keys
            in the return dictionary.
        processes: how many processes to use for file reading.
    Returns:
        dict: file_name (\wo ext) => data dictionary.
    """
    coll = dict()

    def _error(e):
        print(e)

    def _collect_res(res):
        name, data = res
        if uppercase_name:
            name = name.upper()
        coll[name] = data

    pool = Pool(processes)
    for file_path in get_file_paths(folder_path):
        pool.apply_async(load_json_file, args=(file_path, ),
                         callback=_collect_res, error_callback=_error)
    pool.close()
    pool.join()
    return coll


def write_json_file(data, file_path, encoding='utf-8'):
    safe_mkfdir(file_path)
    with open(file_path, encoding=encoding, mode='w') as f:
        json.dump(data, f, indent=2)

#
# def read_json_files(folder_path, uppercase_name=False):
#     """Reads JSON files, assigns the filename (\wo ext) as the key."""
#     coll = dict()
#     for file_path in get_file_paths(folder_path):
#         data = json.load(open(file_path, encoding='utf-8'))
#         name = get_file_name(file_path)
#         if uppercase_name:
#             name = name.upper()
#         coll[name] = data
#     return coll


def iter_json_files(folder_path, uppercase_name=False):
    """Reads JSON files, assigns the filename (\wo ext) as the key."""
    for file_path in get_file_paths(folder_path):
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
        name = get_file_name(file_path)
        if uppercase_name:
            name = name.upper()
        yield name, data


def read_subseqs(input_file_path, sep="\t", nrows=None, order=None,
                 data_type="float", to_tensors=False):
    """Reads floats (codes or features, indxs) from the storage assuming that
    each float
    is space separated. Assumes that each float chunk is separated by `sep`.

     Args:
         input_file_path (str): self-explanatory.
         sep (str): separator of features for different documents.
         nrows (int): that many rows to read from the file.
         order (list): indxs in what order to store floats. Can sub-select a
            subset.
         data_type (str): self-explanatory
         to_tensors (bool): if set, will convert all to Tensors.

    Returns:
        coll (list): list of floats.
    """
    assert data_type in ['int', 'float', 'str']
    type_func = {'float': float, 'int': int, 'str': str}
    tens_func = {'float': T.FloatTensor, 'int': T.LongTensor}

    coll = []
    with open(input_file_path, encoding='utf-8') as f:
        for indx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if nrows and indx > nrows:
                break
            feats = line.split(sep)
            tmp_coll = []
            for _feats in feats:
                _feats = [type_func[data_type](fv) for fv in _feats.strip().split()]
                if order:
                    _feats = [_feats[i] for i in order]
                if len(_feats) == 1:
                    _feats = _feats[0]
                tmp_coll.append(_feats)
            if to_tensors:
                tmp_coll = tens_func[data_type](tmp_coll)
            coll.append(tmp_coll)
    return coll
