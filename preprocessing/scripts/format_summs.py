"""
Splits concatenated summaries to three different files. Writes them to files.
This can be used for separate evaluation of verdicts, pros, and cons.
"""
from argparse import ArgumentParser
from shared_lib.utils.helpers.formatting import format_all_summs
from shared_lib.utils.helpers.io import read_data, write_data
from shared_lib.utils.helpers.paths_and_files import get_file_name, safe_mkdir
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file-path', required=True)
    parser.add_argument('--output-folder-path', required=True)
    args = parser.parse_args()

    input_file_path = args.input_file_path
    output_folder_path = args.output_folder_path
    base_file_name = ".".join(input_file_path.split("/")[-1].split(".")[:-1])

    # paths
    verd_out_file_path = os.path.join(output_folder_path, f'{base_file_name}.verd')
    pros_file_path = os.path.join(output_folder_path, f'{base_file_name}.pros')
    cons_file_path = os.path.join(output_folder_path, f'{base_file_name}.cons')

    # formatting summaries
    summs = read_data(input_file_path)
    verds, pros, cons = format_all_summs(summs)

    # writing to the storage
    safe_mkdir(output_folder_path)
    write_data(verds, verd_out_file_path)
    write_data(pros, pros_file_path)
    write_data(cons, cons_file_path)
