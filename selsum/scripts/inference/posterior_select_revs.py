import torch
from selsum.models.selsum import SelSum
from selsum.criterions.nelbo import NELBO
from selsum.tasks.selsum_task import SelSumTask
from shared_lib.utils.helpers.paths_and_files import safe_mkfdir
from selsum.utils.helpers.data import chunk_data
from shared_lib.utils.helpers.io import read_subseqs, read_data
from torch.cuda import empty_cache
from shared_lib.utils.constants.general import SEQ_SEP
from shared_lib.utils.helpers.logging_funcs import init_logger
import os
from argparse import ArgumentParser


def posterior_select_revs(data_path, checkpoint_path, split='valid',
                          output_folder_path='outputs/q',
                          bart_dir='../artifacts/bart/', ndocs=10, batch_size=20):
    """Tags reviews using the posterior. Saves three types of artifacts:
    1. *.prob: probabilities of selected reviews
    2. *.source: the actual selected reviews (text), formatted for downstream
        summarization
    3. *.tag: binary tags for all reviews, where ones indicate the selected ones.
        This can be used to train the prior.
    """
    tag_sep = " "
    greedy = True

    # paths
    feat_file_path = os.path.join(data_path, f'{split}.feat')
    src_inp_file_path = os.path.join(data_path, f'{split}.source')
    src_out_file_path = os.path.join(output_folder_path, f'{split}.source')
    tag_out_file_path = os.path.join(output_folder_path, f'{split}.tag')
    prob_out_file_path = os.path.join(output_folder_path, f'{split}.prob')

    # features
    feats = read_subseqs(feat_file_path)
    feat_chunks = chunk_data(feats, size=batch_size)

    # source documents
    src = read_data(src_inp_file_path)
    src_chunks = chunk_data(src, size=batch_size)

    imodel = SelSum.posterior_from_pretrained(bpe='gpt2', bart_dir=bart_dir,
                                              checkpoint_file=checkpoint_path,
                                              gpt2_encoder_json=os.path.join(bart_dir, 'encoder.json'),
                                              gpt2_vocab_bpe=os.path.join(bart_dir, 'vocab.bpe'))

    logger.info(f"Features file path: {feat_file_path}")
    logger.info(f"Running inference based on: {checkpoint_path}")

    imodel.cuda()
    imodel.eval()

    count = 0
    chunk_process_count = 0
    print_period = round(len(feat_chunks) / 10)

    safe_mkfdir(src_out_file_path)
    src_out_file = open(src_out_file_path, mode='w', encoding='utf-8')
    tag_out_file = open(tag_out_file_path, mode='w', encoding='utf-8')
    prob_out_file = open(prob_out_file_path, mode='w', encoding='utf-8')

    for src_chunk, feat_chunk in zip(src_chunks, feat_chunks):
        with torch.no_grad():
            doc_indxs, doc_probs = imodel.infer(feats=feat_chunk,
                                                ndocs=ndocs,
                                                greedy=greedy)
        for _docs_indxs, _doc_probs, _src, _feats in zip(doc_indxs, doc_probs, src_chunk, feat_chunk):
            assert len(_src.split(SEQ_SEP)) == len(_feats)

            # source
            _src = [_s.strip() for _s in _src.split(SEQ_SEP)]
            _sel_indxs = [_src[i] for i in _docs_indxs]
            src_out_file.write(f'{SEQ_SEP} '.join(_sel_indxs) + '\n')
            src_out_file.flush()

            # probs
            _doc_probs = [f'{100.*_p:.2f}' for _p in _doc_probs]
            prob_out_file.write(" ".join(_doc_probs) + '\n')
            prob_out_file.flush()

            # tags
            _tags = ["0"] * len(_src)
            for _i in _docs_indxs:
                _tags[_i] = "1"
            tag_out_file.write(tag_sep.join(_tags) + '\n')
            tag_out_file.flush()

        count += 1

        chunk_process_count += len(feat_chunk)
        if print_period > 0 and (count % print_period == 0):
            empty_cache()
            logger.info(f"Processed {chunk_process_count}/{len(feats)} lines")\

    src_out_file.close()
    tag_out_file.close()
    prob_out_file.close()

    logger.info(f"Processed {chunk_process_count}/{len(feats)} lines")
    logger.info(f"Output is saved to: {output_folder_path}")


if __name__ == '__main__':
    logger = init_logger("")
    parser = ArgumentParser()
    parser.add_argument('--data-path', required=True,
                        help="Location of FAIRSEQ (not binarized) data")
    parser.add_argument('--split', default='valid')
    parser.add_argument('--checkpoint-path', type=str,
                        help="Path to the model checkpoint")
    parser.add_argument('--output-folder-path', required=True, default='outputs/q')
    parser.add_argument('--bart-dir', default='../artifacts/bart/')
    parser.add_argument('--ndocs', type=int, default=10,
                        help="The number of documents to tag as 1.")
    parser.add_argument('--batch-size', type=int, default=20)
    posterior_select_revs(**vars(parser.parse_args()))
