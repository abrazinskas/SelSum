import torch
from selsum.models.prior import Prior
from selsum.tasks.doc_tagging_task import DocTaggingTask
from selsum.criterions.multi_tagging import MultiTagging
from shared_lib.utils.helpers.paths_and_files import safe_mkfdir
from selsum.utils.helpers.data import chunk_data
from shared_lib.utils.helpers.io import read_data
from shared_lib.utils.helpers.logging_funcs import init_logger
from torch.cuda import empty_cache
from shared_lib.utils.constants.general import SEQ_SEP
from argparse import ArgumentParser
import os


def prior_select_revs(data_path, checkpoint_path, split='valid',
                      output_folder_path='outputs/p', bart_dir='../artifacts/bart/',
                      ndocs=10, batch_size=10):
    """Selects reviews using the prior."""
    # paths
    src_inp_file_path = os.path.join(data_path, f'{split}.source')
    src_out_file_path = os.path.join(output_folder_path, f'{split}.source')
    tag_out_file_path = os.path.join(output_folder_path, f'{split}.tags')
    src = read_data(src_inp_file_path)

    imodel = Prior.from_pretrained(
        bart_dir=bart_dir,
        checkpoint_file=checkpoint_path,
        gpt2_encoder_json=os.path.join(bart_dir, 'encoder.json'),
        gpt2_vocab_bpe=os.path.join(bart_dir, 'vocab.bpe'),
        bpe='gpt2', strict=True
    )

    imodel.cuda()
    imodel.eval()
    imodel.half()

    src_chunks = chunk_data(src, size=batch_size)

    count = 0
    chunk_process_count = 0
    print_period = round(len(src_chunks) / 100)

    safe_mkfdir(src_out_file_path)
    out_src_file = open(src_out_file_path, mode='w', encoding='utf-8')
    out_tag_file = open(tag_out_file_path, mode='w', encoding='utf-8')

    logger.info(f"Selecting {ndocs} reviews for each data point "
                f"from {src_inp_file_path}.")
    logger.info(f"Inference based on: {checkpoint_path}")

    for src_chunk in src_chunks:
        with torch.no_grad():
            # TODO: please note that the number of tags can be different from the
            # TODO: number of initial reviews because of filtering on the
            # TODO: dataset side
            tags, docs = imodel.infer(src_chunk, top_k=ndocs,
                                      out_seq_sep=f'{SEQ_SEP} ')

        for _tags, _docs in zip(tags, docs):
            out_tag_file.write(_tags + '\n')
            out_src_file.write(_docs + '\n')
            out_src_file.flush()
            out_tag_file.flush()

        count += 1
        chunk_process_count += len(src_chunk)
        empty_cache()
        if print_period > 0 and (count % print_period == 0):
            logger.info(f"Processed {chunk_process_count}/{len(src)} lines")

    out_src_file.close()
    out_tag_file.close()

    logger.info(f"Processed {chunk_process_count}/{len(src)} lines")
    logger.info(f"Output is saved to: {output_folder_path}")


if __name__ == '__main__':
    logger = init_logger("")
    parser = ArgumentParser()
    parser.add_argument('--data-path', required=True,
                        help="Location of FAIRSEQ (not binarized) data")
    parser.add_argument('--split', default='valid')
    parser.add_argument('--checkpoint-path', type=str,
                        help="Path to the model checkpoint")
    parser.add_argument('--output-folder-path', required=True,
                        default='output/p')
    parser.add_argument('--bart-dir', default='../artifacts/bart/')
    parser.add_argument('--ndocs', type=int, default=10,
                        help="The number of documents to select")
    parser.add_argument('--batch-size', type=int, default=20)
    prior_select_revs(**vars(parser.parse_args()))
