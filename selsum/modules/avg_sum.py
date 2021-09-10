import torch as T
from torch import nn


class AvgSum(nn.Module):
    """Averages the input features."""

    def forward(self, x, x_mask=None):
        """
        Args:
            x (FloatTensor): input features
                (batch_size, seq_len, dim)
            x_mask (BooleanTensor): True is set at positions that should be masked
                (batch_size, seq_len)

        Returns:
            out (FloatTensor): [batch_size, input_dim]
            norm_weights (FloatTensor): [batch_size, seq_len]
        """
        scores = T.ones_like(x_mask)
        scores = scores.masked_fill_(x_mask, 0)
        norm_weights = scores / (~x_mask).sum(-1, keepdim=True)

        out = (x * norm_weights).sum(dim=1)

        return out
