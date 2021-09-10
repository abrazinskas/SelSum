from argparse import ArgumentParser
from shared_lib.utils.helpers.io import iter_json_files, write_data
from shared_lib.utils.constants.aggregates import WEBSITE_SUMMS, CUSTOMER_REVIEWS,\
    VERDICT, PROS, CONS
from shared_lib.utils.constants.general import TRAIN_PART, VAL_PART, TEST_PART
from shared_lib.utils.constants.general import SEQ_SEP
from shared_lib.utils.helpers.logging_funcs import init_logger
from logging import getLogger
import os


logger = getLogger(__name__)


def fairseq_format(input_folder_path, output_folder_path, sep_symb=SEQ_SEP):
    """Converts JSON aggregated files to src and tgt files that can be used as
    input to FAIRSeq models. Tgt will contain concatenated verdicts, pros and
    cons, concatenated by `sep_symb`.

    Reviews are joined by `sep_symb`. Individual pros/cons bullet points are
    joined with '.', and the separation between pros and cons is indicated by
    `sep_symb`. For example, the string might look as follows:

        verdict </s> pro1 . pro2 . pro3 </s> con1 . con2 .

    Assumes partitions of train, val, and test sets in `input_folder_path`. Also,
    assumes that each input entry has a complete set of summaries: verdicts, pros,
    and cons.
    """
    for part_name in [TRAIN_PART, VAL_PART, TEST_PART]:

        # collectors
        src_coll = []
        tgt_coll = []
        asin_coll = []

        # file paths
        inp_part_folder_path = os.path.join(input_folder_path, part_name)
        src_file_path = os.path.join(output_folder_path, f'{part_name}.source')
        tgt_file_path = os.path.join(output_folder_path, f'{part_name}.target')
        asin_file_path = os.path.join(output_folder_path, f'{part_name}.asin')

        for asin_name, du in iter_json_files(inp_part_folder_path):
            docs = []
            docs += [rev['text'] for rev in du[CUSTOMER_REVIEWS]]
            doc_str = f'{sep_symb} '.join(docs)

            for summ in du[WEBSITE_SUMMS]:
                verd, pros, cons = summ[VERDICT], summ[PROS], summ[CONS]
                form_pros = " . ".join(pros) + " ."
                form_cons = " . ".join(cons) + " ."

                if len(pros) and len(cons) and len(verd):
                    tgt_str = f'{verd}{sep_symb} {form_pros}{sep_symb} {form_cons}'
                    src_coll.append(doc_str)
                    tgt_coll.append(tgt_str)
                    asin_coll.append(asin_name)

        # DUMPING TO THE STORAGE
        write_data(src_coll, src_file_path)
        write_data(tgt_coll, tgt_file_path)
        write_data(asin_coll, asin_file_path)
        logger.info(f'Wrote {len(tgt_coll)} instances for '
                    f'\'{part_name}\' partition')


if __name__ == '__main__':
    init_logger("")
    parser = ArgumentParser()
    parser.add_argument('--input-folder-path', required=True,
                        help="The directory path with JSON ASINs.")
    parser.add_argument('--output-folder-path', required=True,
                        help="The output directory for FAIRSEQ partitions.")
    parser.add_argument('--sep-symb', default=SEQ_SEP, type=str)
    fairseq_format(**vars(parser.parse_args()))
