import logging
from fairseq.tasks import FairseqTask, register_task
from selsum.data.doc_tagging_dataset import DocTaggingDataset
from selsum.data.seq_splitter import SeqSplitter
from selsum.data.seq_dataset import SeqDataset
from selsum.data.src_wrapper import SrcWrapper
from selsum.data.sep_replacer import SepReplacer
from selsum.data.doc_count_reducer import DocCountReducer
from selsum.utils.helpers.io import read_tags, make_bin_path
from selsum.utils.helpers.model import setup_bpe
from shared_lib.utils.constants.general import SEQ_SEP
from fairseq.data import (Dictionary, data_utils, StripTokenDataset,
                          TruncateDataset)
import numpy as np
import os
import torch as T

logger = logging.getLogger(__name__)
MAX_DOC_LEN = 280


@register_task('doc_tagging_task')
class DocTaggingTask(FairseqTask):
    """Document tagging task where a model is trained to predict binary tags.
    
    This task is used to train the prior that selects a subset of informative 
    reviews from a large collection.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--data', help='path to data directories.')
        parser.add_argument('--tag-path', help='path to directory with tags.')
        parser.add_argument("--bart-dir", type=str, required=True,
                            help="Directory path to load the BPE encoder to obtain"
                                 "subword ids of `sent-sep`.")
        parser.add_argument(
            '--max-source-positions', default=1024, type=int, metavar='N',
            help='max number of tokens in the source sequence')
        parser.add_argument('--min-docs', default=10, type=int,
                            help="Removes instances that have less documents "
                                 "than specified. The warning displayed will be "
                                 "misleading, see `doc_tagging_dataset.py` for more information.")
        parser.add_argument('--seq-sep', type=str, default=SEQ_SEP,
                            help='Separation symbol between sentences.')
        parser.add_argument("--shuffle", action="store_true")
        parser.add_argument("--sep-to-replace",
                            help='Separation symbol between docs that needs to '
                                 'be replaced. Used for sentence extraction.')
        parser.add_argument('--max-doc-len', type=int, default=MAX_DOC_LEN)

    def __init__(self, args, dict):
        super().__init__(args)
        self.args = args
        self.dictionary = dict
        # adding mask to avoid conflicts
        self.mask_indx = self.dictionary.add_symbol('<mask>')
        self.bpe = setup_bpe(args.bart_dir)
        self.seq_sep = args.seq_sep
        self.seq_sep_indxs = [dict.index(s) for s in
                              self.bpe.encode(self.seq_sep).split()]
        self.pad_indx = self.dictionary.pad()
        self.bos_indx = self.dictionary.bos()
        self.eos_indx = self.dictionary.eos()

        self.sep_to_replace = args.sep_to_replace
        if args.sep_to_replace:
            self.sep_to_replace_indxs = [dict.index(s) for s in
                                         self.bpe.encode(self.sep_to_replace).split()]
        else:
            self.sep_to_replace_indxs = None

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        base_path = self.args.data
        if self.args.tag_path is not None:
            base_tag_path = self.args.tag_path
        else:
            base_tag_path = base_path
        bin_path = make_bin_path(base_path)
        src_path = os.path.join(bin_path, f"{split}.source-target.source")
        tgt_path = os.path.join(base_tag_path, f"{split}.tag")

        tgt = read_tags(tgt_path)
        tgt = [T.LongTensor(t) for t in tgt]
        tgt = SeqDataset(tgt, np.array([len(t) for t in tgt]))

        src_dataset = data_utils.load_indexed_dataset(src_path, self.dictionary)
        if src_dataset is None:
            raise ValueError("Could not load the source dataset.")
        src_dataset = self._create_seq_dataset(src_dataset)
        ds = DocTaggingDataset(src=src_dataset, min_docs=self.args.min_docs,
                               tgt=tgt, pad_indx=self.pad_indx,
                               eos_indx=self.eos_indx, mask_indx=self.mask_indx,
                               shuffle=self.args.shuffle,
                               max_doc_len=self.args.max_doc_len)

        logger.info("Split: {0}, Loaded {1} samples".format(split, len(ds)))
        logger.info(f"Tag path: {tgt_path}")
        logger.info(f"The dataset size: {len(ds)}")

        self.datasets[split] = ds

    def build_dataset_for_inference(self, src_tokens, src_lengths,
                                    max_doc_len=MAX_DOC_LEN):
        ds = SeqDataset(src_tokens, src_lengths)
        ds = self._create_seq_dataset(ds)
        ds = DocTaggingDataset(src=ds, tgt=None, pad_indx=self.pad_indx,
                               eos_indx=self.eos_indx, mask_indx=self.mask_indx,
                               max_doc_len=max_doc_len, shuffle=False)
        return ds

    def _create_seq_dataset(self, dataset):
        """Helper method for creating a source dataset."""
        dataset = StripTokenDataset(dataset, self.eos_indx)
        if self.sep_to_replace_indxs:
            dataset = SepReplacer(dataset, sep_indxs=self.sep_to_replace_indxs,
                                  repl_indx=self.seq_sep_indxs)
        dataset = SeqSplitter(dataset=dataset, sep_indxs=self.seq_sep_indxs)
        dataset = SrcWrapper(dataset=dataset, bos_indx=self.bos_indx,
                             eos_indx=self.eos_indx)
        return dataset

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        dictionary = Dictionary.load(os.path.join(args.bart_dir, 'dict.txt'))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def max_positions(self):
        return self.args.max_source_positions
