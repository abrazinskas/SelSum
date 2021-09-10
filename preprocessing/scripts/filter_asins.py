from argparse import ArgumentParser
from shared_lib.utils.helpers.logging_funcs import init_logger
from logging import getLogger
from multiprocessing import Pool
from shared_lib.utils.constants.aggregates import CUSTOMER_REVIEWS
from shared_lib.utils.helpers.general import is_in_range
from shared_lib.utils.helpers.io import iter_json_files, write_json_file
import os
from copy import copy

logger = getLogger(__name__)


def _error_callback(e):
    logger.error(f'{type(e).__name__}: {e}')


def _filter(du, output_file_path, min_rev_len, max_rev_len, min_revs, max_revs):
    """Filters a passed ASIN (`du`)."""
    revs = du[CUSTOMER_REVIEWS]
    before_filt_rev_count = len(revs)
    revs = [rev for rev in revs
            if is_in_range(len(rev['text'].split()), min_rev_len, max_rev_len)]
    if revs:
        if min_revs is not None and len(revs) < min_revs:
            logger.info(f"Skipping {os.path.basename(output_file_path)}. "
                        f"Review count after/before: "
                        f"{len(revs)}/{before_filt_rev_count} filtering.")
            return
        if max_revs is not None:
            revs = revs[:max_revs]
        du[CUSTOMER_REVIEWS] = revs
        write_json_file(du, output_file_path)


def filter_asins(input_folder_path, output_folder_path, nworkers, **kwargs):
    """Filters entries in raw JSON files based on review statistics. Discards
    ASINs that have less than a minimum number of reviews.
    """
    logger.info(f"Parallel workers: {nworkers}")
    pool = Pool(nworkers)
    for asin, du in iter_json_files(input_folder_path):
        kwgs = copy(kwargs)
        kwgs['output_file_path'] = os.path.join(output_folder_path, f'{asin}.json')
        kwgs['du'] = du
        pool.apply_async(_filter, error_callback=_error_callback, kwds=kwgs)
    pool.close()
    pool.join()


if __name__ == '__main__':
    init_logger("")
    parser = ArgumentParser()
    parser.add_argument('--input-folder-path', required=True,
                        help="Path to input raw ASIN files.")
    parser.add_argument('--output-folder-path', required=True,
                        help="Path where to store filtered ASIN files.")
    parser.add_argument('--min-rev-len', type=int, default=10,
                        help="The minimum review length.")
    parser.add_argument('--max-rev-len', type=int,
                        help="The maximum review length.")
    parser.add_argument('--min-revs', type=int,
                        help="The minimum number of reviews.")
    parser.add_argument('--max-revs', type=int,
                        help="If set will take the first maximum reviews.")
    parser.add_argument('--nworkers', type=int, default=10)
    filter_asins(**vars(parser.parse_args()))
