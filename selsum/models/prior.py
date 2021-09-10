import logging
import torch
from fairseq.models import register_model, BaseFairseqModel, \
    register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from torch.nn import Module, Sequential, Tanh, Linear, LayerNorm, ReLU, Dropout
from selsum.interfaces.prior_interface import PriorInterface
from selsum.utils.helpers.architectures import bart_encoder_architecture,\
    contextualizer_architecture
from fairseq.models.transformer import TransformerEncoder, \
    DEFAULT_MAX_SOURCE_POSITIONS, TransformerModel, Embedding
from fairseq import utils, checkpoint_utils
from selsum.utils.helpers.model import freeze_params, copy_module_params, update_names
from selsum.modules.weighted_sum import WeightedSum
from selsum.utils.helpers.args import get_prefixed_args
from selsum.modules.contextualizer import Contextualizer
from copy import copy
import numpy as np

logger = logging.getLogger(__name__)
DEFAULT_MAX_TARGET_POSITIONS = 100  # the maximum collection size


@register_model('prior')
class Prior(BaseFairseqModel):
    """Combines a standard transformer encoder that encodes individual reviews

    1. Transformer encoder: encodes individual reviews
    2. Pooler (weighted sum): aggregates states
    3. Projector (feed-forward): yields a score for each review
    """

    def __init__(self, args, encoder, pooler, contxt, projector):
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.contxt = contxt
        self.projector = projector
        self.apply(init_bert_params)

    def forward(self, src_tokens, src_lengths, group_src_indxs):
        group_src_indxs_mask = group_src_indxs == -1

        # individual review encoding
        encoder_out = self.encoder(src_tokens=src_tokens,
                                   src_lengths=src_lengths)
        y = encoder_out.encoder_out.transpose(1, 0)
        padding_mask = encoder_out.encoder_padding_mask

        y = self.pooler(y, padding_mask)

        # grouping pooled review states
        y = y[group_src_indxs]

        # contextualization
        if self.contxt is not None:
            y = self.contxt(y, group_src_indxs_mask)

        # projecting
        logits = self.projector(y).squeeze(-1)

        logits[group_src_indxs_mask] = - np.inf

        return logits

    def get_logits(self, x):
        return x

    @staticmethod
    def add_args(parser):
        super(Prior, Prior).add_args(parser)
        bart_encoder_architecture(parser)
        parser.add_argument('--proj-hidden-dim', type=int, default=150)
        parser.add_argument('--proj-nlayers', type=int, default=2)
        parser.add_argument('--proj-dropout', type=float, default=0.1)
        parser.add_argument('--ws-hidden-dim', type=int, default=150)
        parser.add_argument('--ws-dropout', type=float, default=0.1)
        parser.add_argument('--contxt-encoder-layers', type=int, default=2)
        parser.add_argument('--proj-nclasses', default=3, type=int)

    @classmethod
    def build_model(cls, args, task):
        """Builds a new model instance."""
        args.ws_input_dim = args.encoder_embed_dim
        encoder = cls.build_encoder(args, task)
        pooler = cls.build_pooler(args, task)

        if args.contxt_encoder_layers > 0:
            contxt = cls.build_contextualizer(args, task)
        else:
            contxt = None

        projector = cls.build_projector(args, task)
        return cls(args, encoder=encoder, pooler=pooler, contxt=contxt,
                   projector=projector)

    @classmethod
    def build_encoder(cls, args, task):
        """Creates the base encoder model."""
        if getattr(args, "tagger_max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        src_dict = task.source_dictionary
        embed_tokens = TransformerModel.build_embedding(
            args, src_dict, args.encoder_embed_dim, args.encoder_embed_path)
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_pooler(self, args, task):
        """Builds a pooling mechanism that aggregates states."""
        return WeightedSum(args)

    @classmethod
    def build_projector(self, args, task):
        """Builds a projection module that converts the state to a logit."""
        hidden_dim = args.proj_hidden_dim
        input_dim = args.encoder_embed_dim
        dropout = args.proj_dropout
        proj_classes = args.proj_nclasses

        proj_ffnn = Sequential()
        for n in range(1, args.proj_nlayers+1):
            proj_ffnn.add_module(f"lin-{n}", Linear(input_dim, hidden_dim))
            proj_ffnn.add_module(f"non-lin-{n}", ReLU())
            proj_ffnn.add_module(f"dropout-{n}", Dropout(dropout))
            input_dim = hidden_dim
        proj_ffnn.add_module(f"lnorm", LayerNorm(hidden_dim))
        proj_ffnn.add_module(f'lin-out', Linear(input_dim, proj_classes))

        return proj_ffnn

    @classmethod
    def build_contextualizer(self, args, task):
        """Builds a contextualizer module."""
        contx = Contextualizer(get_prefixed_args(args, "contxt"))
        return contx

    @classmethod
    def from_pretrained(cls, checkpoint_file, **kwargs):
        models, args, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_file], arg_overrides=kwargs)
        return PriorInterface(args, task, models[0])

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions()

    def upgrade_state_dict_named(self, state_dict, name):
        """Removes the bart decoder from the state dictionary.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        update_names(self, name, state_dict)

        # deleting the unused modules from the state
        to_delete = ['q_network', 'decoder']
        for to_del in to_delete:
            for k in list(state_dict.keys()):
                if k.startswith(f'{to_del}.'):
                    del state_dict[k]

        # adding modules if not present in the checkpoint
        modules_to_copy = ['pooler', 'projector']
        if self.contxt is not None:
            modules_to_copy.append('contxt')
            state_dict['contxt.version'] = torch.Tensor([1])
        copy_module_params(self, state_dict, modules_to_copy)


@register_model_architecture('prior', 'prior')
def prior_architecture(args):
    bart_encoder_architecture(args)
    # pooler
    args.ws_hidden_dim = getattr(args, 'ws_hidden_dim', 150)
    args.ws_dropout = getattr(args, 'ws_dropout', 0.1)
    # contextualizer
    args.contxt_encoder_embed_dim = getattr(args, 'encoder_embed_dim')
    contextualizer_architecture(args)
    # projector
    args.proj_nlayers = getattr(args, 'proj_nlayers', 2)
    args.proj_hidden_dim = getattr(args, 'proj_hidden_dim', 150)
    args.proj_dropout = getattr(args, 'proj_dropout', 0.1)
    args.proj_nclasses = getattr(args, 'proj_nclasses', 3)
