from copy import copy
import logging
import torch
import torch.nn as nn
from typing import List
from fairseq import utils
from fairseq.data import encoders
from shared_lib.utils.helpers.data import sent_splitter_multi, concat


logger = logging.getLogger(__name__)


class SumInterface(nn.Module):
    """Basic interface for summary generation."""

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model
        self.bpe = encoders.build_bpe(args)
        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))
        self.sep_symb = self.task.sep_symb
        self.sep_indxs = self.task.sep_indxs
        self.tgt_repl_sep_indx = self.task.tgt_repl_sep_indx

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, docs):
        """Prepares the documents string for decoding by tokenizing it.

        Args:
            docs (str): concatenated by a separator documents.

        Returns:
            LongTensor with tokens.
        """
        docs = self.bpe.encode(docs)
        tokens = self.task.source_dictionary.encode_line(docs, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        """Converts tokens to the human readable format. Sentence agnostic."""
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        # remove start and end symbols
        if tokens[0] == self.task.target_dictionary.bos():
            tokens = tokens[1:]
        if tokens[-1] == self.task.target_dictionary.eos():
            tokens = tokens[:-1]
        res = self.bpe.decode(self.task.source_dictionary.string(tokens))
        res = res.strip()
        return res

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        dataset = self.task.build_dataset_for_inference(src_tokens,
            [x.numel() for x in src_tokens])
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    def sample(self, docs: List[str], beam: int = 1, verbose: bool = False,
               **kwargs) -> str:
        input = [self.encode(_docs) for _docs in docs]
        hypos = self.generate(tokens=input, beam=beam, verbose=verbose, **kwargs)

        x_tokens = []
        for x in hypos:
            toks = x['tokens']
            subseqs = sent_splitter_multi(toks, [self.tgt_repl_sep_indx],
                                          incl_split_symb=False)
            toks = concat(subseqs, self.sep_indxs)
            x_tokens.append(toks)

        return [self.decode(toks) for toks in x_tokens]

    def generate(self, tokens: List[torch.LongTensor], beam: int = 5,
                 verbose: bool = False, **kwargs) -> torch.LongTensor:
        sample = self._build_sample(src_tokens=tokens)

        # build generator using current args as well as any kwargs
        gen_args = copy(self.args)
        setattr(gen_args, 'beam', beam)
        for k, v in kwargs.items():
            setattr(gen_args, k, v)

        generator = self.task.build_generator([self.model], gen_args)
        summs = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=sample['net_input']['src_tokens']
                .new_zeros((len(tokens), 1))
                .fill_(self.task.target_dictionary.bos()),
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        # Process top predictions
        hypos = [x[0] for x in summs]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos



