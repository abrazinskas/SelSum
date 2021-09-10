import torch as T
from fairseq.data import encoders
from shared_lib.utils.helpers.topk_ngram_blocker import topk_with_ngram_blocker
import numpy as np
from fairseq import utils
from typing import List


class PriorInterface(T.nn.Module):
    """Interface for selecting review subsets via the trained prior."""

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model
        self.bpe = encoders.build_bpe(args)
        # this is useful for determining the device
        self.register_buffer('_float_tensor', T.tensor([0], dtype=T.float))
        self.seq_sep = self.task.seq_sep

    def encode(self, docs):
        """Prepares the documents string for decoding by BPE tokenizing it.

        Args:
            docs (str): concatenated by a separator documents.

        Returns:
            LongTensor with tokens.
        """
        docs = self.bpe.encode(docs)
        tokens = self.task.source_dictionary.encode_line(docs, append_eos=False)
        return tokens.long()

    def decode(self, tokens: T.LongTensor):
        """Converts tokens to the human readable format. Sentence agnostic."""
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        tokens = [t for t in tokens if t not in
                  [self.task.bos_indx, self.task.eos_indx, self.task.pad_indx]]
        res = self.bpe.decode(self.task.source_dictionary.string(tokens))
        res = res.strip()
        return res

    def infer(self, docs, **kwargs):
        input = [self.encode(_docs) for _docs in docs]
        output = self.predict(input, **kwargs)
        return output

    def predict(self, docs, top_k=3, out_seq_sep=None):
        """Tags documents.

        Please note that the number of tags can be different from the
        initial number of reviews.

        Args:
            docs (list): document strings.
            top_k (int): that many sequences to select per entry.
            ngram_block (int): if passed, will block from being selected n-gram
                overlapping sequences with already selected ones.

        Returns:
            tag_coll (list): list of binary tags (str).
            sel_doc_coll (list): corresponding documents (str).
        """
        sample = self._build_sample(docs)
        group_src_indxs = sample['net_input']['group_src_indxs']
        src = sample['net_input']['src_tokens']
        logits = self.model(**sample['net_input'])

        # converting to numpy
        probs = T.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
        # re-creating docs, as some might have been filtered out
        docs = [[self.decode(src[i]) for i in _group_indxs if i != -1]
                for _group_indxs in group_src_indxs]

        group_src_indxs_mask = (group_src_indxs == -1).cpu().numpy()

        tag_coll = []
        sel_doc_coll = []

        for _docs, _probs, _mask in zip(docs, probs, group_src_indxs_mask):
            assert len(_probs) == len(_mask)

            # selecting only unmasked probs
            _probs = [l for l, m in zip(_probs, _mask) if not m]

            if len(_probs) <= top_k:
                _top_k_indxs = range(len(_probs))
            else:
                _top_k_indxs = np.argsort(_probs)[::-1][:top_k]

            _top_k_indxs = sorted(_top_k_indxs)
            # docs
            _sel_docs = [_docs[i] for i in _top_k_indxs]
            _sel_docs = self._format_output(_sel_docs, out_seq_sep)
            sel_doc_coll.append(_sel_docs)

            # tags
            _tags = [0] * len(_docs)
            for i in _top_k_indxs:
                _tags[i] = 1
            _tags = self._format_output(_tags, sep=" ")
            tag_coll.append(_tags)

        return tag_coll, sel_doc_coll

    def _format_output(self, seq, sep):
        return sep.join([str(s).strip() for s in seq])

    def _build_sample(self, src_tokens: List[T.LongTensor], **kwargs):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
            **kwargs
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    @property
    def device(self):
        return self._float_tensor.device
