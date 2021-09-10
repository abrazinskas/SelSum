from shared_lib.utils.constants.general import TRAIN_PART, VAL_PART, TEST_PART
from shared_lib.utils.helpers.io import read_data
import os


def read_fairseq_data(data_folder_path):
    """Reads the partition data. Returns a dictionary with (src, tgt) tuples."""
    coll = dict()
    for part_name in [TRAIN_PART, VAL_PART, TEST_PART]:
        src_file_path = os.path.join(data_folder_path, f'{part_name}.source')
        tgt_file_path = os.path.join(data_folder_path, f'{part_name}.target')
        src = read_data(src_file_path, encoding='utf-8')
        tgt = read_data(tgt_file_path, encoding='utf-8')
        coll[part_name] = (src, tgt)
    return coll
