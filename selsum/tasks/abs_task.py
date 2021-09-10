import logging
import os
import torch as T
import numpy as np
from selsum.data.abs_dataset import AbsDataset
from selsum.data.doc_reducer import DocReducer
from selsum.data.sep_replacer import SepReplacer
from selsum.data.seq_splitter import SeqSplitter
from selsum.data.seq_dataset import SeqDataset
from selsum.data.src_wrapper import SrcWrapper
from selsum.utils.helpers.io import get_bin_path
from fairseq.data import (PrependTokenDataset, AppendTokenDataset, Dictionary,
    data_utils, StripTokenDataset)
from selsum.utils.constants.model import SEP_REPL
from fairseq.tasks import FairseqTask, register_task
from selsum.utils.helpers.model import setup_bpe

MAX_DOC_LEN = 280

logger = logging.getLogger(__name__)


@register_task('abs_task')
class AbsTask(FairseqTask):
    """Abstractive summarization task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--data', help='path to data directories')
        parser.add_argument("--bart-dir", type=str, required=True,
                            help="Directory path to load the BPE encoder to obtain"
                                 "subword ids of `sent-sep`.")
        parser.add_argument(
            '--max-source-positions', default=1024, type=int, metavar='N',
            help='max number of tokens in the source sequence')
        parser.add_argument(
            '--max-target-positions', default=1024, type=int, metavar='N',
            help='max number of tokens in the target sequence')
        parser.add_argument('--sep-symb', type=str,
                            help='Separation symbol between sentences.'
                                 ' E.g.," </s>".', default=" </s>")
        parser.add_argument("--shuffle", action="store_true")

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.args = args
        self.dictionary = dictionary
        self.pad_indx = dictionary.pad()
        self.eos_indx = dictionary.eos()
        # adding mask to avoid conflicts
        self.mask_indx = self.dictionary.add_symbol('<mask>')
        self.bpe = setup_bpe(args.bart_dir)
        self.sep_symb = args.sep_symb
        self.sep_indxs = [dictionary.index(s) for s in
                          self.bpe.encode(self.sep_symb).split()]
        # 'tgt_repl_sep_indx' is used to replace the separator in targets
        # such that the model has to predict only one sub-word instead of
        # multiple
        self.tgt_repl_sep_indx = self.dictionary.index(SEP_REPL)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        dictionary = Dictionary.load(os.path.join(args.bart_dir, 'dict.txt'))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        bin_split_path = get_bin_path(self.args.data, epoch=epoch)
        src_path = os.path.join(bin_split_path, f'{split}.source-target.source')
        tgt_path = os.path.join(bin_split_path, f'{split}.source-target.target')

        tgt_ds = self._load_target_dataset(tgt_path)

        # reducing the number of reviews to avoid problems with FAIRSEQ batching
        offset = 60
        budgets = [self.args.max_tokens - s - offset for s in tgt_ds.sizes]
        src_ds = data_utils.load_indexed_dataset(src_path,
                                                 dataset_impl=self.args.dataset_impl,
                                                 dictionary=self.dictionary)
        if src_ds is None:
            raise ValueError(f"Could not load the source dataset in "
                             f"'{src_path}'.")
        src_ds = self._create_source_dataset(dataset=src_ds,
                                             max_doc_len=MAX_DOC_LEN,
                                             dataset_sizes=src_ds.sizes,
                                             max_budgets=budgets)

        ds = AbsDataset(src=src_ds, tgt=tgt_ds,
                        shuffle=self.args.shuffle,
                        pad_indx=self.dictionary.pad(),
                        bos_indx=self.dictionary.bos(),
                        eos_indx=self.dictionary.eos(),
                        mask_indx=self.mask_indx)
        logger.info("Split: {0}, Loaded {1} samples".format(split, len(ds)))
        logger.info(f"The dataset size: {len(ds)}")
        self.datasets[split] = ds

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src = self._create_source_dataset(src_tokens, src_lengths,
                                          max_doc_len=MAX_DOC_LEN)
        return AbsDataset(src=src, tgt=None, shuffle=False,
                          pad_indx=self.dictionary.pad(),
                          bos_indx=self.dictionary.bos(),
                          eos_indx=self.dictionary.eos(),
                          mask_indx=self.mask_indx)

    def _create_source_dataset(self, dataset, dataset_sizes, max_budgets=None,
                               max_doc_len=None):
        """Properly wraps the source dataset. If no features are provided,
        will simply split sub-sequences.
        """
        dataset = SeqDataset(dataset, dataset_sizes)
        dataset = StripTokenDataset(dataset, self.dictionary.eos())
        dataset = SeqSplitter(dataset=dataset, sep_indxs=self.sep_indxs)
        if max_budgets is not None or max_doc_len is not None:
            dataset = DocReducer(dataset=dataset, max_budgets=max_budgets,
                                 sort_docs=False, max_doc_len=max_doc_len)
        dataset = SrcWrapper(dataset=dataset, bos_indx=self.dictionary.bos(),
                             eos_indx=self.dictionary.eos())
        return dataset

    def _load_target_dataset(self, path):
        """Loads the target dataset from the storage. Wraps it properly."""
        dataset = data_utils.load_indexed_dataset(path, self.dictionary)
        if dataset is None:
            raise ValueError(f"Could not load the target dataset in '{path}'.")
        dataset = SeqDataset(dataset, dataset.sizes)
        dataset = StripTokenDataset(dataset, self.dictionary.eos())
        dataset = PrependTokenDataset(dataset, self.dictionary.bos())
        dataset = AppendTokenDataset(dataset, self.dictionary.eos())
        dataset = SepReplacer(dataset=dataset, sep_indxs=self.sep_indxs,
                              repl_indx=self.tgt_repl_sep_indx)
        return dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


