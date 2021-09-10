from fairseq.tasks import FairseqTask, register_task
from selsum.tasks.abs_task import AbsTask, MAX_DOC_LEN
from fairseq.data.raw_label_dataset import RawLabelDataset
import torch as T
import numpy as np
from fairseq.data.data_utils import load_indexed_dataset
from selsum.utils.helpers.io import make_bin_path, get_bin_path
from shared_lib.utils.helpers.io import read_subseqs
from selsum.data.infer_dataset import InferDataset
from shared_lib.utils.constants.features import FEATURE_ORDER
import os
import logging
from copy import copy
from selsum.utils.helpers.subsampling import subsample
from selsum.utils.constants.model import SEL_STEP, SUM_STEP


logger = logging.getLogger(__name__)


@register_task('selsum_task')
class SelSumTask(AbsTask):
    """First the approximate posterior is used to select a subset of reviews and
    subsequently these reviews are summarized.

    Two steps are separate for computational efficiency.
    """

    @staticmethod
    def add_args(parser):
        AbsTask.add_args(parser)
        parser.add_argument('--sel-sample-num', default=1, type=int,
                            help="The number of samples to use to approximate "
                                 "the reconstruction term when the selector is "
                                 "trained.")
        parser.add_argument('--sum-sample-num', default=1, type=int,
                            help="The number of samples to use to approximate "
                                 "the reconstruction term when the summarizer is "
                                 "trained.")
        parser.add_argument('--sel-step-num', type=int, default=1,
                            help='The number of times to optimize the selector '
                                 'for each batch.')
        parser.add_argument('--sum-step-num', type=int, default=1,
                            help='The number of times to optimize the summarizer '
                                 'for each batch.')
        parser.add_argument('--bline-sample-num', type=int, default=1,
                            help='The number of samples to use to approximate '
                                 'the baseline.')
        parser.add_argument('--ndocs', type=int, default=20,
                            help='The maximum number of reviews/documents to '
                                 'sample per instance.')

        parser.add_argument('--debug', action='store_true')

    def __init__(self, args, dictionary):
        super(SelSumTask, self).__init__(args, dictionary)
        self.ndocs = args.ndocs
        setattr(args, 'q_encoder_feature_dim', len(FEATURE_ORDER))
        self.sel_sample_num = args.sel_sample_num
        self.sum_sample_num = args.sum_sample_num
        self.sel_step_num = args.sel_step_num
        self.sum_step_num = args.sum_step_num
        self.bline_sample_num = args.bline_sample_num

    def _sel_step(self, sample, model, criterion):
        """Computes the loss for the selector with the frozen summarizer."""

        # computing the baseline
        b_sample = subsample(distr_model=model.q_network, sample=sample,
                             rsample_num=self.bline_sample_num,
                             sample_type="prior", ndocs=self.ndocs,
                             pad_indx=self.pad_indx, eos_indx=self.eos_indx,
                             mask_idx=self.mask_indx)
        with T.no_grad():
            _, _, b_log_out = criterion(model, b_sample, reinforce=False,
                                        sample_num=self.bline_sample_num)
            bline = b_log_out['cll'].unsqueeze(-1).repeat(1, self.sel_sample_num).view(-1)

        # the actual step
        sample = subsample(distr_model=model.q_network, sample=sample,
                           rsample_num=self.sel_sample_num, ndocs=self.ndocs,
                           pad_indx=self.pad_indx, eos_indx=self.eos_indx,
                           mask_idx=self.mask_indx, sample_type='posterior')
        loss, samp_size, log_out = criterion(model=model, sample=sample,
                                             reinforce=True, baseline=bline,
                                             sample_num=self.sel_sample_num)
        return loss, samp_size, log_out

    def _sum_step(self, sample, model, criterion):
        """Computes the loss for the summarizer with the frozen selector."""
        sample = subsample(distr_model=model.q_network, sample=sample,
                           rsample_num=1, ndocs=self.ndocs,
                           pad_indx=self.pad_indx, eos_indx=self.eos_indx,
                           mask_idx=self.mask_indx, sample_type='posterior')
        loss, samp_size, log_out = criterion(model=model, sample=sample,
                                             reinforce=False, sample_num=1)
        return loss, samp_size, log_out

    def train_step(self, sample, model, criterion, optimizer, update_num,
                   ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)

        if SEL_STEP in sample and sample[SEL_STEP]:
            loss, samp_size, log_out = self._sel_step(sample=sample, model=model,
                                                      criterion=criterion)
        else:
            loss, samp_size, log_out = self._sum_step(sample=sample, model=model,
                                                      criterion=criterion)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        return loss, samp_size, log_out

    def valid_step(self, sample, model, criterion):
        model.eval()
        with T.no_grad():
            loss, smpl_size, logging_out = self._sum_step(sample, model, criterion)
        return loss, smpl_size, logging_out

    def load_dataset(self, split, epoch=1, **kwargs):
        # paths
        base_path = self.args.data
        bin_path = make_bin_path(base_path)
        src_path = os.path.join(bin_path, f"{split}.source-target.source")
        tgt_path = os.path.join(bin_path, f"{split}.source-target.target")
        feats_path = os.path.join(base_path, f"{split}.feat")

        # loading
        src_ds = load_indexed_dataset(src_path, dictionary=self.dictionary)
        if src_ds is None:
            raise ValueError(f"Could not load the source dataset in "
                             f"'{src_path}'.")
        src_ds = self._create_source_dataset(dataset=src_ds,
                                             dataset_sizes=src_ds.sizes)
        tgt_ds = self._load_target_dataset(tgt_path)

        # features
        feat_ds = read_subseqs(feats_path, data_type='float', to_tensors=False,
                               order=list(range(len(FEATURE_ORDER))))
        feat_ds = RawLabelDataset(feat_ds)
        ds = InferDataset(src_ds=src_ds, feat_ds=feat_ds, tgt_ds=tgt_ds,
                          max_doc_len=MAX_DOC_LEN, debug=self.args.debug)
        logger.info("Split: {0}, Loaded {1} samples".format(split, len(ds)))
        logger.info(f"The dataset size: {len(ds)}")
        self.datasets[split] = ds

    def build_dataset_for_inference(self, feats, **kwargs):
        raise NotImplementedError

