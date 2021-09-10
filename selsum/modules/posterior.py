from selsum.modules.ff_encoder import FFEncoder
import torch.nn as nn
from selsum.modules.linear_ar_decoder import LinearARDecoder


DEFAULT_MAX_POSITIONS = 100


class Posterior(nn.Module):
    """An encoder-decoder model for selecting reviews guided by the summary."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, feats, feats_mask, prev_sel_indxs, **kwargs):
        """# TODO: explain what each argument is
        """
        encoder_out = self.encoder(feats, feats_mask)
        y = self.decoder(prev_sel_indxs, encoder_out=encoder_out)
        return y

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if getattr(args, "decoder_max_positions", None) is None:
            args.decoder_max_positions = DEFAULT_MAX_POSITIONS
        encoder = FFEncoder(args)
        decoder = LinearARDecoder(args)
        return cls(encoder, decoder)

