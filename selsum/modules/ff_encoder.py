from fairseq.models import FairseqEncoder
from fairseq.models.transformer import LayerNorm
from torch.nn import Sequential
import torch.nn as nn
from torch import Tensor
from typing import Dict
from fairseq.models.fairseq_encoder import EncoderOut


class FFEncoder(FairseqEncoder):
    """Simple feed-forward encoder that transforms features. Wraps the output
    with the proper Encoder object.
    """

    def __init__(self, args):
        super().__init__(dictionary=None)
        hidden_dim = args.encoder_hidden_dim
        nlayers = args.encoder_nlayers
        dropout = args.encoder_dropout
        input_dim = args.encoder_feature_dim

        ffnn = Sequential()
        for n in range(1, nlayers+1):
            ffnn.add_module(f"lin-{n}", nn.Linear(input_dim, hidden_dim))
            ffnn.add_module(f"non-lin-{n}", nn.Tanh())
            ffnn.add_module(f"dropout-{n}", nn.Dropout(dropout))
            input_dim = hidden_dim
        ffnn.add_module(f"lnorm", LayerNorm(hidden_dim))
        self.ffnn = ffnn

    def forward(self, feats, feats_mask):
        """
        Args:
            feats (FloatTensor): (bsz, seq_len, feature_dim)
            feats_mask (BoolTensor): (bsz, seq_len)
        """
        return EncoderOut(
            encoder_out=self.ffnn(feats),  # B x T x C
            encoder_padding_mask=feats_mask,  # B x T
            encoder_embedding=None, encoder_states=None, src_tokens=None,
            src_lengths=None)

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out: Dict[str, Tensor] = {}
        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        return EncoderOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B x C
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B x T
            encoder_embedding=None, encoder_states=None, src_tokens=None,
            src_lengths=None,
        )
