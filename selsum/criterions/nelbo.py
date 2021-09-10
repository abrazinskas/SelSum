from fairseq import metrics, utils
import math
from fairseq.criterions import register_criterion, FairseqCriterion
import torch as T
from shared_lib.utils.helpers.general import merge_dicts, flatten
from logging import getLogger

EPS = 1e-7


logger = getLogger(__name__)


def compute_ll(lprobs, target, ignore_index=None, reduce=True):
    """Computes the log-likelihood.

    Args:
        lprobs: [batch_size, seq_len, vocab_size]
        target: [batch_size, vocab_size]
        ignore_index: what index in `target` to ignore.
        reduce: whether to sum over the `seq_len`.

    Returns:
        ll: [batch_size] or [batch_size, seq_len]
    """
    bs = lprobs.size(0)

    lprobs = lprobs.view(-1, lprobs.size(-1))
    target = target.view(-1, 1)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    ll = lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        ll.masked_fill_(pad_mask, 0.)
    ll = ll.squeeze(-1).view(bs, -1)
    ll = ll.sum(-1) if reduce else ll
    return ll


@register_criterion('nelbo')
class NELBO(FairseqCriterion):
    """Computes the negative expectation lower bound with the reconstruction
    term that has REINFORCE re-formulation.
    """

    def compute_reinforce_loss(self, cll, weights, baseline):
        """Calculates negative reconstruction term re-formulated with REINFORCE.

        Args:
            cll: conditional log-likelihood in the reconstruction term.
                [batch_size]
            weights: log q(r_k) probability that is used for REINFORCE.
                [batch_size, k]
            baseline: baseline values in the reconstruction term used to
                reduce the gradient estimate variance.
                [batch_size]
        """
        rec = weights.sum(-1) * (cll - baseline).detach()
        loss = - rec
        return loss

    def forward(self, model, sample, reinforce=True, sample_num=1,
                baseline=None):
        p_distr, distr_mask = model.get_prior(sample)
        bsz = p_distr.size(0)

        if reinforce:
            assert baseline is not None
            q_distr = model.comp_posterior(**sample['net_input'])
            weights = model.comp_weights(q_distr, **sample['net_input'],
                                         mean=False)

            with T.no_grad():
                gen_output = model(**sample['net_input'])
        else:
            with T.no_grad():
                q_distr = model.comp_posterior(**sample['net_input'])
            gen_output = model(**sample['net_input'])

        target = model.get_targets(sample, gen_output)
        pred_lprobs = model.get_normalized_probs(gen_output, log_probs=True)

        # computing main components in the ELBO
        cll = compute_ll(lprobs=pred_lprobs, target=target, reduce=True,
                         ignore_index=self.padding_idx)

        # normalizing by the seq. length (technically the whole ELBO equation)
        seq_len = (target != self.padding_idx).sum(-1)
        cll = cll / seq_len

        if reinforce:
            loss = self.compute_reinforce_loss(cll=cll, weights=weights,
                                               baseline=baseline)
        else:
            loss = - cll

        # aggregation over samples that were used to approximate ELBO for the
        # the same summary
        real_bs = int(bsz / sample_num)
        cll = cll.view(real_bs, sample_num).mean(-1)
        loss = loss.view(real_bs, sample_num).mean(-1)

        sample_size = len(loss)
        logging_output = {'loss': loss.data, 'cll': cll.data,
                          'ntokens': sample['ntokens'],
                          'nsentences': sample['target'].size(0),
                          'sample_size': sample_size}

        loss_sum = loss.sum()

        return loss_sum, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss = flatten(list(log.get('loss')) for log in logging_outputs)
        cll = flatten(list(log.get('cll')) for log in logging_outputs)
        ppl = [T.exp(-_ll) for _ll in cll]

        # averaging
        avg_loss = sum(loss) / len(loss)
        avg_cll = sum(cll) / len(cll)
        avg_ppl = sum(ppl) / len(ppl)

        # storing to metrics
        metrics.log_scalar('loss', avg_loss, len(loss), round=3)
        metrics.log_scalar('cll', avg_cll, len(cll), round=3)
        metrics.log_scalar('ppl', avg_ppl, len(ppl), round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
