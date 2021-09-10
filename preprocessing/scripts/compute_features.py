from argparse import ArgumentParser
from logging import getLogger
from multiprocessing import Pool
from shared_lib.utils.helpers.logging_funcs import init_logger
from shared_lib.utils.helpers.io import read_data
from shared_lib.utils.constants.general import TRAIN_PART, VAL_PART
from shared_lib.utils.helpers.general import sort_hash, flatten
from shared_lib.utils.helpers.paths_and_files import safe_mkfdir
from preprocessing.utils.compute import get_features
from preprocessing.utils.io import read_fairseq_data
from preprocessing.utils.general import format_feats
from preprocessing.utils.compute import get_max_lens
import os
logger = getLogger(__name__)


RES_COLL = {}
TO_ROUGE_MEASURE = {'f': 'fmeasure', 'p': 'precision', 'r': 'recall'}


def _error_callback(e):
    logger.error(f'{type(e).__name__}: {e}')


def _work(indx, **kwargs):
    feats = get_features(**kwargs)
    return indx, feats


def _success_callback(res):
    indx, scores = res
    if indx in RES_COLL:
        logger.error(f"Duplicate key {indx}")
    RES_COLL[indx] = scores


def compute_features(data_folder_path, lex_file_path=None, nworkers=16):
    """Computes and dumps features for source-target pairs in FAIRSEQ files.

    Using a number of heuristics, computes in parallel features for data
    in each partition (training, validation, and testing) of FAIRSEQ files.
    Features are computed separately for pros&cons and verdicts.

    Writes calculated features to the partition corresponding file with the
    `.feat` extension.

    These features are used as input to the approximate posterior (q).
    """
    logger.info(f"Parallel workers: {nworkers}")
    data = read_fairseq_data(data_folder_path)
    src_max_len, verd_max_len, pc_max_len = get_max_lens(
        src=flatten([data[TRAIN_PART][0], data[VAL_PART][0]]),
        tgt=flatten([data[TRAIN_PART][1], data[VAL_PART][1]])
    )
    lex = set(read_data(lex_file_path))
    logger.info(f"Read data for all partitions")
    logger.info(f'Maximum lens: '
                f'(src: {src_max_len}, verd: {verd_max_len}, pc: {pc_max_len})')

    for part_name, (src, tgt) in data.items():
        global RES_COLL
        RES_COLL = {}
        logger.info(f"Computing features for: '{part_name}' partition")
        out_file_path = os.path.join(data_folder_path, f'{part_name}.feat')
        assert len(src) == len(tgt)
        pool = Pool(nworkers)
        for indx, (_src, _tgt) in enumerate(zip(src, tgt)):
            kwgs = {'src': _src, 'tgt': _tgt, 'lex': lex, 'indx': indx,
                    'src_max_len': src_max_len, 'verd_max_len': verd_max_len,
                    'pc_max_len': pc_max_len}
            pool.apply_async(func=_work, error_callback=_error_callback,
                             callback=_success_callback, kwds=kwgs)
        pool.close()
        pool.join()
        logger.info(f"Dumping features to: '{out_file_path}'")

        # dumping to the storage
        assert len(RES_COLL) == len(src)
        safe_mkfdir(out_file_path)
        with open(out_file_path, encoding='utf-8', mode='w') as f:
            for _, _feats in sort_hash(RES_COLL):
                f.write(format_feats(_feats) + "\n")


if __name__ == '__main__':
    init_logger("")
    parser = ArgumentParser()
    parser.add_argument('--data-folder-path', required=True,
                        help="The directory of FAIRSEQ formatted files.")
    parser.add_argument('--lex-file-path', required=True,
                        help="Aspect lexicon file path.")
    parser.add_argument('--nworkers', default=16, type=int,
                        help="The number of parallel workers to compute "
                             "features.")
    compute_features(**vars(parser.parse_args()))
