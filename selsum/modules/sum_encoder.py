from fairseq.models.transformer import TransformerEncoder
from fairseq.models.fairseq_encoder import NamedTuple, Tensor, Optional, List
import torch
from torch.nn import Linear, ReLU, Sequential

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Tensor),  # B x T
        ("encoder_embedding", Tensor),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
        ("code_proj", Optional[Tensor])  # B x T
    ],
)


class SumEncoder(TransformerEncoder):
    """Runs the transformer encoder; concats states based on `group_src_indxs`.
    Optionally concatenate control codes to the encoded states.
    """
    def __init__(self, args, src_dict, embed_tokens):
        super(SumEncoder, self).__init__(args, src_dict, embed_tokens)
        if hasattr(args, 'code_names') and args.code_names is not None:
            self.code_proj = Sequential()
            self.code_proj.add_module("lin1", Linear(len(args.code_names),
                                                     args.encoder_embed_dim))
            self.code_proj.add_module("non-lin", ReLU())
            self.code_proj.add_module('lin2', Linear(args.encoder_embed_dim,
                                                     args.encoder_embed_dim))

    def forward(self, src_tokens, src_lengths, group_src_indxs, codes=None,
                return_all_hiddens: bool = False):
        enc_out = super(SumEncoder, self).forward(src_tokens=src_tokens,
                                                  src_lengths=src_lengths,
                                                  return_all_hiddens=return_all_hiddens)
        enc_out = self.group_enc_out(enc_out, group_src_indxs)
        if codes is not None:
            enc_out = self.add_codes(enc_out, codes)
        return enc_out

    @torch.jit.unused
    def forward_non_torchscript(self, net_input):
        codes = net_input['codes'] if 'codes' in net_input else None
        return self.forward(src_tokens=net_input['_src_tokens'],
                            src_lengths=net_input['_src_lengths'],
                            group_src_indxs=net_input['_group_src_indxs'],
                            codes=codes)

    def group_enc_out(self, enc_out_obj, group_indxs):
        """Groups and concatenates the encoder output based on `group_indxs`."""
        enc_out = enc_out_obj.encoder_out
        enc_pad_mask = enc_out_obj.encoder_padding_mask
        enc_emb = enc_out_obj.encoder_embedding

        seq_len = enc_out.size(0)
        group_nr, el_per_group = group_indxs.shape
        cat_seq_len = el_per_group * seq_len

        # transformations
        enc_out = enc_out.transpose(1, 0)
        enc_out = enc_out[group_indxs].view(group_nr, cat_seq_len, -1)
        enc_out = enc_out.transpose(1, 0)
        enc_emb = enc_emb[group_indxs].view(group_nr, cat_seq_len, -1)
        enc_pad_mask = enc_pad_mask[group_indxs].view(group_nr, -1)

        out = EncoderOut(encoder_out=enc_out,
                         encoder_padding_mask=enc_pad_mask,
                         encoder_embedding=enc_emb,
                         encoder_states=enc_out_obj.encoder_states,
                         src_tokens=enc_out_obj.src_tokens,
                         src_lengths=enc_out_obj.src_lengths,
                         code_proj=None)
        return out

    def add_codes(self, enc_out_obj, codes):
        """Adds projected codes to the embeddings."""
        code_proj = self.code_proj(codes)
        encoder_out = code_proj.unsqueeze(0) + enc_out_obj.encoder_out
        out = EncoderOut(encoder_out=encoder_out,
                         encoder_padding_mask=enc_out_obj.encoder_padding_mask,
                         encoder_embedding=enc_out_obj.encoder_embedding,
                         encoder_states=enc_out_obj.encoder_states,
                         src_tokens=enc_out_obj.src_tokens,
                         src_lengths=enc_out_obj.src_lengths,
                         code_proj=code_proj)
        return out
