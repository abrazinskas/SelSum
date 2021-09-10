from torch import Tensor
from typing import Any, Dict, Optional
import math
from selsum.utils.helpers.model import create_blocker_mask
from fairseq.modules import PositionalEmbedding
import torch as T
import torch.nn as nn


class LinearARDecoder(nn.Module):
    """A simple linear auto-regressive decoder where previous selections/samples
    are blocked from being selected at subsequent steps. The decoder has no
    state as such and logits are re-used.

    Uses the incremental state to cache logits.
    """

    def __init__(self, args):
        super(LinearARDecoder, self).__init__()
        self.score_func = nn.Linear(args.decoder_embed_dim, 1)
        self.embed_positions = PositionalEmbedding(args.decoder_max_positions,
                                                   args.decoder_embed_dim,
                                                   padding_idx=1,
                                                   learned=args.decoder_learned_pos)

    def forward(self, prev_sel_indxs, encoder_out,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None
                ):
        """
        Args:
            prev_sel_indxs: previously selected indxs. The first one should be
                dummy.
                (bsz, seq_len)
            encoder_out: encoded representation of reviews.

        Returns:
            scores: logits
                (bsz, seq_len, vocab_size)
        """

        enc_out = encoder_out.encoder_out
        padding_mask = encoder_out.encoder_padding_mask
        bsz, slen = prev_sel_indxs.shape
        vocab_size = enc_out.size(1)

        if incremental_state is None or len(incremental_state) == 0:
            x = enc_out + self.embed_positions(padding_mask.long())
            scores = self.score_func(x).squeeze(-1)
            scores.masked_fill_(padding_mask, -math.inf)
            prev_sel_bmask = create_blocker_mask(prev_sel_indxs=prev_sel_indxs,
                                                 vocab_size=vocab_size)
        else:
            scores = incremental_state['scores']
            prev_sel_bmask = incremental_state['prev_sel_block_mask']
            # blocking the previously selected indxs
            prev_sel_bmask[T.arange(bsz).unsqueeze(-1), T.arange(slen),
                           prev_sel_indxs] = True

        # update the cache
        if incremental_state is not None:
            incremental_state['scores'] = scores
            incremental_state['prev_sel_block_mask'] = prev_sel_bmask

        # producing auto-regressive scores by 'blocking' previously selected
        # data points
        scores = scores.unsqueeze(1).repeat((1, slen, 1))
        scores = scores.masked_fill_(prev_sel_bmask, -math.inf)

        return scores
