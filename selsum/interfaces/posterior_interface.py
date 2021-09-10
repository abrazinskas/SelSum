import logging
import torch as T
from fairseq.data import encoders
from selsum.utils.posterior_generator import PosteriorGenerator
from selsum.utils.helpers.collators import collate_features
from selsum.utils.helpers.subsampling import sample_from_q
from fairseq.utils import apply_to_sample
from selsum.utils.constants.model import FEATS

logger = logging.getLogger(__name__)


class PosteriorInterface(T.nn.Module):
    """Posterior interface that selects reviews based on features. These features
    are computed based on reviews and product summary.
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model
        self.bpe = encoders.build_bpe(args)
        # this is useful for determining the device
        self.register_buffer('_float_tensor', T.tensor([0], dtype=T.float))

    @property
    def device(self):
        return self._float_tensor.device

    def infer(self, feats, ndocs, **kwargs):
        """Runs the inference network. Returns selected
        formatted document indices and
        """
        coll_sizes = [len(f) for f in feats]

        sample = self._build_sample([{FEATS: f} for f in feats])
        doc_indxs, q_probs = sample_from_q(self.model, sample=sample,
                                           sample_size=ndocs, **kwargs)

        # sorting by document indxs as the encoder is order agnostic
        bsz = doc_indxs.size(0)
        sort_indxs = T.argsort(doc_indxs, dim=-1)
        doc_indxs = doc_indxs[T.arange(bsz).unsqueeze(-1), sort_indxs]
        q_probs = q_probs[T.arange(bsz).unsqueeze(-1), sort_indxs]

        form_doc_indxs = self._format_output(doc_indxs, coll_sizes)
        form_q_probs = self._format_output(q_probs, coll_sizes)

        return form_doc_indxs, form_q_probs

    def _format_output(self, entries, coll_sizes):
        """Removes padded entries. Converts to list of numpy arrays."""
        entries = entries.cpu().numpy()
        coll = []
        for _entry, _coll_size in zip(entries, coll_sizes):
            _entry = _entry[:_coll_size]
            coll.append(_entry)
        return coll

    def _build_sample(self, feats_sample):
        """Builds a sample for running the posterior network."""
        sample = collate_features(feats_sample, add_dummy=True)
        sample = apply_to_sample(lambda tensor: tensor.to(self.device), sample)
        return sample
