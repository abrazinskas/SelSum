import torch as T
import torch.nn as nn
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from selsum.modules.linear_ar_decoder import LinearARDecoder
from fairseq.models.transformer import TransformerEncoder, TransformerModel
from selsum.modules.ff_encoder import FFEncoder
from selsum.modules.avg_sum import AvgSum


DEFAULT_MAX_POSITIONS = 100


class Prior(nn.Module):

    def __init__(self, args, base_enc, pooler, encoder, decoder):
        """
        Args:
            base_enc: extracts continues features from the input documents.
            encoder: linear encoder
        """
        super().__init__()
        self.base_enc = base_enc
        self.pooler = pooler
        self.encoder = encoder
        self.decoder = decoder
        self.apply(init_bert_params)
        self.freeze_base_enc = args.freeze_base_enc

    def forward(self, src_tokens, src_lengths, src_mask, group_src_indxs,
                group_src_indxs_mask, prev_sel_indxs):
        # individual review feature extraction
        if self.freeze_base_enc:
            with T.no_grad():
                encoder_out = self.base_enc(src_tokens=src_tokens,
                                            src_lengths=src_lengths)
        else:
            encoder_out = self.base_enc(src_tokens=src_tokens,
                                        src_lengths=src_lengths)
        y = encoder_out.encoder_out.transpose(1, 0)  # [bsz, seq_len, dim]
        # pooling
        y = self.pooler(y, src_mask)
        # concatenating together pooled states
        y = y[group_src_indxs]
        # encoder-decoder
        encoder_out = self.encoder(y, group_src_indxs_mask)
        y = self.decoder(prev_sel_indxs, encoder_out=encoder_out)
        return y

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if getattr(args, "decoder_max_positions", None) is None:
            args.decoder_max_positions = DEFAULT_MAX_POSITIONS
        base_enc = cls.build_base_enc(args, task)
        encoder = FFEncoder(args)
        decoder = LinearARDecoder(args)
        return cls(args, base_enc=base_enc, pooler=AvgSum(), encoder=encoder,
                   decoder=decoder)

    @classmethod
    def build_base_enc(cls, args, task):
        """Transformer encoder that can be pre-init. with fine-tuned params."""
        src_dict = task.source_dictionary
        embed_tokens = TransformerModel.build_embedding(
            args, src_dict, args.encoder_embed_dim, args.encoder_embed_path)
        base_enc = TransformerEncoder(args, src_dict, embed_tokens)
        return base_enc
