import logging
import torch as T
from fairseq.models import register_model, register_model_architecture
from fairseq import checkpoint_utils
from fairseq.models.transformer import base_architecture, \
    DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS
from selsum.models.sum import Sum, sum_base_architecture, init_bert_params
from selsum.utils.helpers.architectures import posterior_architecture
from selsum.interfaces.sum_interface import SumInterface
from selsum.interfaces.posterior_interface import PosteriorInterface
from selsum.modules.posterior import Posterior
from selsum.utils.helpers.computation import softmax
from selsum.utils.helpers.args import get_prefixed_args
from selsum.utils.helpers.model import copy_module_params, update_names

logger = logging.getLogger(__name__)
EPS = 1e-6


@register_model('selsum')
class SelSum(Sum):
    """Combines the posterior to select a review subset from the collection and
    subsequently summarize it.
    """

    @staticmethod
    def add_args(parser):
        Sum.add_args(parser=parser)
        # adding selector specific arguments
        parser.add_argument('--q-nlayers', type=int, default=2)
        parser.add_argument('--q-encoder-hidden-dim', type=int, default=150)
        parser.add_argument('--q-encoder-nlayers', type=int, default=1)
        parser.add_argument('--q-encoder-dropout', type=float, default=0.1)

    def __init__(self, args, encoder, decoder, q_network):
        super(Sum, self).__init__(args, encoder, decoder)
        self.q_network = q_network
        self.apply(init_bert_params)

    def forward(self, src_tokens, src_lengths, _group_src_indxs,
                _src_tokens, _src_lengths, prev_output_tokens, **kwargs):
        """Runs the encoder-decoder summarizer."""
        encoder_out = self.encoder(_src_tokens, src_lengths=_src_lengths,
                                   group_src_indxs=_group_src_indxs)
        y, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out,
                                features_only=False)
        return y, extra

    def comp_posterior(self, feats, feats_mask, prev_sel_indxs, **kwargs):
        """Runs the inference network, computes the posterior distribution."""
        q_scores = self.q_network(feats=feats, feats_mask=feats_mask,
                                  prev_sel_indxs=prev_sel_indxs)
        q_distr = softmax(q_scores, dim=-1)
        return q_distr

    def comp_weights(self, q_distr, sel_indxs, sel_indxs_mask, mean=False,
                     **kwargs):
        """Returns REINFORCE log q() weights. Optionally normalized by the
        number of selected reviews.
        """
        q_sel_probs = q_distr[T.arange(sel_indxs.size(0)).unsqueeze(-1),
                              T.arange(sel_indxs.size(1)), sel_indxs]
        wts = (q_sel_probs + EPS).log().masked_fill_(sel_indxs_mask, 0.)
        if mean:
            wts = (wts / (~sel_indxs_mask).sum(-1, keepdims=True)).sum(-1)
        return wts

    def upgrade_state_dict_named(self, state_dict, name):
        """Adds the current state of the inference network to the dictionary."""
        assert state_dict is not None
        update_names(self, name, state_dict)
        copy_module_params(self, state_dict, ['q_network'])

    @classmethod
    def from_pretrained(cls, checkpoint_file, **kwargs):
        """Returns the summarization interface."""
        models, args, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_file], arg_overrides=kwargs)
        return SumInterface(args, task, models[0])

    @classmethod
    def posterior_from_pretrained(cls, checkpoint_file, **kwargs):
        """Returns the posterior/selector interface."""
        models, args, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_file], arg_overrides=kwargs)
        return PosteriorInterface(args, task, models[0].q_network)

    @classmethod
    def build_model(cls, args, task):
        """Builds a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        q_args = get_prefixed_args(args, prefix="q")
        q_network = Posterior.build_model(q_args, task)
        return cls(args, encoder, decoder, q_network)

    @staticmethod
    def get_prior(sample):
        """Returns the prior and its mask."""
        p_distr = sample['net_input']['p_distr']
        distr_mask = sample['net_input']['sel_indxs_mask']
        return p_distr, distr_mask


@register_model_architecture('selsum', 'selsum')
def selsum_base_architecture(args):
    sum_base_architecture(args)
    posterior_architecture(args)
