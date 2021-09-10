import torch
import torch.nn as nn
from fairseq.models.transformer import TransformerEncoder as TEnc, LayerNorm
from torch.nn import Module
from fairseq.modules.positional_embedding import PositionalEmbedding
import torch.nn.functional as F
from fairseq import utils


class Contextualizer(Module):
    """Performs sequence contextualization. Expects as input continuous
    representations instead of tokens.
    """

    def __init__(self, args):
        super().__init__()
        self.register_buffer("version", torch.Tensor([1]))

        self.dropout = args.dropout
        self.embed_dim = args.encoder_embed_dim
        self.max_source_positions = args.max_source_positions

        self.embed_positions = PositionalEmbedding(num_embeddings=self.max_source_positions,
                                                   embedding_dim=self.embed_dim,
                                                   padding_idx=1,
                                                   learned=args.learned_pos)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_output:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        return TEnc.build_encoder_layer(self, args)

    def forward(self, x, x_mask):
        """
        Args:
            x (FloatTensor): sentence representations.
            x_mask (BoolTensor): padded elements are indicated by ``True``.
                shape: `(batch, src_len)`
        Returns:
            encoder_out (Tensor): the last encoder layer's output of
            shape `(batch, src_len, embed_dim)`
        """
        # pretend that all token ids are 0, while the padding_idx is 1

        x = x + self.embed_positions(x_mask.long())
        x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, x_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)

        return x
