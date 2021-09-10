import torch
from selsum.models.selsum import SelSum
import os
from shared_lib.utils.helpers.paths_and_files import safe_mkdir
from selsum.utils.helpers.data import chunk_data
from selsum.criterions.nelbo import NELBO
from selsum.tasks.abs_task import AbsTask
from torch.cuda import empty_cache
from shared_lib.utils.helpers.logging_funcs import init_logger
from shared_lib.utils.helpers.formatting import format_all_summs
from shared_lib.utils.helpers.io import read_data, write_data
from shared_lib.utils.helpers.stats import print_subseq_text_stats
from argparse import ArgumentParser


def gen_summs(data_path, checkpoint_path,
              output_folder_path='outputs/summs', split='valid',
              bart_dir='../artifacts/bart/', batch_size=10):
    """Generates summaries."""

    # paths
    safe_mkdir(output_folder_path)
    src_file_path = os.path.join(data_path, f'{split}.source')
    hypo_file_path = os.path.join(output_folder_path, f'{split}.hypo')
    verd_out_file_path = os.path.join(output_folder_path, f'{split}.verd')
    pros_out_file_path = os.path.join(output_folder_path, f'{split}.pros')
    cons_out_file_path = os.path.join(output_folder_path, f'{split}.cons')

    # beam search hyperparams
    min_len = 75
    lenpen = 5
    max_len_a = 0
    max_len_b = 600
    beam_size = 5

    model = SelSum.from_pretrained(bart_dir=bart_dir, task="abs_task",
                                   checkpoint_file=checkpoint_path, bpe='gpt2',
                                   gpt2_encoder_json=os.path.join(bart_dir,
                                                                  'encoder.json'),
                                   gpt2_vocab_bpe=os.path.join(bart_dir,
                                                               'vocab.bpe'))

    logger.info(f'Beam hyper-parameters: '
                f'beam_size={beam_size} | min_len={min_len} | lenpen={lenpen} | max_len_a={max_len_a} |'
                f' max_len_b={max_len_b}')
    logger.info(f"Source: {src_file_path}")
    logger.info(f"Running inference based on: {checkpoint_path}")

    model.cuda()
    model.eval()
    model.half()

    # reading data
    src = read_data(src_file_path)
    src_chunks = chunk_data(src, size=batch_size)

    count = 0
    chunk_process_count = 0
    print_period = round(len(src_chunks) / 100)

    hypo_coll = []

    with open(hypo_file_path, 'w', encoding='utf-8') as fout:
        for i in range(len(src_chunks)):
            src_chunk = src_chunks[i]
            with torch.no_grad():
                hypo_batch = model.sample(docs=src_chunk, beam=beam_size,
                                          lenpen=lenpen, min_len=min_len,
                                          no_repeat_ngram_size=3,
                                          max_len_a=max_len_a,
                                          max_len_b=max_len_b)
            for hypo in hypo_batch:
                fout.write(hypo + '\n')
                fout.flush()
            count += 1
            chunk_process_count += len(src_chunk)
            hypo_coll += hypo_batch
            empty_cache()
            if print_period > 0 and (count % print_period == 0):
                logger.info(f"Processed: {chunk_process_count}/{len(src)} lines")

    logger.info(f"Processed: {chunk_process_count}/{len(src)} lines")
    logger.info(f"Output is saved to: {output_folder_path}")

    # performing formatting to make it easier to later evaluate
    verds, pros, cons = format_all_summs(summs=hypo_coll,
                                         empty_symb=".")
    write_data(verds, verd_out_file_path)
    write_data(pros, pros_out_file_path)
    write_data(cons, cons_out_file_path)


if __name__ == '__main__':
    logger = init_logger("")
    parser = ArgumentParser()
    parser.add_argument('--data-path', required=True,
                        help="Location of FAIRSEQ (not binarized) data")
    parser.add_argument('--bart-dir', default='../artifacts/bart/')
    parser.add_argument('--split', default='valid')
    parser.add_argument('--checkpoint-path', type=str,
                        help="Path to the model checkpoint")
    parser.add_argument('--output-folder-path', required=True,
                        default='output/p')
    parser.add_argument('--batch-size', type=int, default=20)
    gen_summs(**vars(parser.parse_args()))
