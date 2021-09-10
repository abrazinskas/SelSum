import torch as T
from torch.nn.functional import softmax


def masked_softmax(scores, mask, dim=-1):
    """Normalizes scores using masked softmax operation.
    # TODO: make it value based instead of reference?

    Args:
        scores (FloatTensor): unnormalized scores.
        mask (BoolTensor): binary mask where True indicates positions to mask.
        dim: dimensionality over which to perform normalization.
    """
    assert scores.shape == mask.shape
    scores = scores.masked_fill(mask, float('-inf'))
    probs = softmax(scores, dim=dim)
    return probs


def masked_normalize(scores, mask, dim=-1):
    """
    # TODO: make it value based instead of reference?

    Args:
        scores ():
        mask ():
        dim ():
    """
    scores = scores.masked_fill(mask, 0.)
    norm = scores.sum(dim=dim, keepdim=True)
    probs = scores / norm
    return probs


def cat_cross_entropy(q, p, mask, dim=-1, eps=1e-8):
    """Calculates CE[q||p] = E_{q}[log p] assuming both are Categorical."""
    ce = q * T.log(p + eps)
    ce[mask] = 0.
    ce = ce.sum(dim=dim)
    return ce


def cat_kld(q, p, eps=1e-8, reduce=False):
    """Calculated categorical Kullback-Leibler divergence."""
    # kld = q * (T.log(q + eps) - T.log(p + eps))
    # q[T.isclose(q, T.zeros_like(q))] = eps
    # p[T.isclose(p, T.zeros_like(p))] = eps
    # kld = q * (T.log(q) - T.log(p))
    kld = q * (T.log(q + eps) - T.log(p + eps))
    if reduce:
        kld = kld.sum(-1)
    return kld


# def cat_kld(q, p, mask=None, dim=-1, eps=1e-8):
#     """Calculated categorical Kullback-Leibler divergence."""
#     kld = q * (T.log(q + eps) - T.log(p + eps))
#     if mask is not None:
#         kld[mask] = 0.
#     kld = kld.sum(dim=dim)
#     return kld
