from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, \
    TransformerDecoder as Decoder
from selsum.modules.sum_encoder import SumEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.bart.model import bart_base_architecture
from fairseq import checkpoint_utils
from selsum.interfaces.sum_interface import SumInterface
from selsum.utils.helpers.model import update_names
import logging


logger = logging.getLogger(__name__)


@register_model('sum')
class Sum(TransformerModel):
    """Encoder-decoder summarizer based on the Transformer architecture."""

    def __init__(self, args, encoder, decoder):
        super(Sum, self).__init__(args, encoder, decoder)
        self.apply(init_bert_params)

    def forward(self, src_tokens, src_lengths, _group_src_indxs,
                _src_tokens, _src_lengths, prev_output_tokens):
        """
        Args:
            src_tokens: concatenated reviews (passed for consistency),
                the size matches the number of target summaries.
            src_lengths: same as `src_tokens`.
            _group_src_indxs:
            _src_tokens:
            _src_lengths:
            prev_output_tokens:

        """
        encoder_out = self.encoder(_src_tokens, src_lengths=_src_lengths,
                                   group_src_indxs=_group_src_indxs)
        y, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out,
                                features_only=False)
        return y, extra

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return SumEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return Decoder(args, tgt_dict, embed_tokens)

    @classmethod
    def from_pretrained(cls, checkpoint_file, **kwargs):
        models, args, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_file], arg_overrides=kwargs)
        return SumInterface(args, task, models[0])

    def upgrade_state_dict_named(self, state_dict, name):
        update_names(self, name, state_dict)


@register_model_architecture('sum', 'sum_base')
def sum_base_architecture(args):
    bart_base_architecture(args)
