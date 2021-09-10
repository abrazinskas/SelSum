from torch.nn import Module, Sequential, ReLU, Linear, LayerNorm, Dropout
from torch.nn.functional import softmax
import math


class WeightedSum(Module):
    """Computes the weighted sum of the input (e.g., encoder states). Weighting
    is performed using a multi-layer perceptron.
    """

    def __init__(self, args):
        super(WeightedSum, self).__init__()

        hidden_dim = args.ws_hidden_dim
        input_dim = args.ws_input_dim
        dropout = args.ws_dropout

        self.ffnn = Sequential()
        self.ffnn.add_module("lin-1", Linear(input_dim, hidden_dim))
        self.ffnn.add_module("non-lin-1", ReLU())
        self.ffnn.add_module("dropout-1", Dropout(dropout))
        self.ffnn.add_module('lin-2', Linear(hidden_dim, 1))

    def forward(self, x, x_mask=None):
        """
        Args:
            x (FloatTensor): [batch_size, seq_len, input_dim]
            x_mask (BoolTensor): [batch_size, seq_len]

        Returns:
            out (FloatTensor): [batch_size, input_dim]
        """
        scores = self.ffnn(x).squeeze(-1)
        if x_mask is not None:
            scores = scores.masked_fill_(x_mask, -math.inf)
        norm_weights = softmax(scores, dim=1)
        out = (x * norm_weights.unsqueeze(-1)).sum(dim=1)
        return out
