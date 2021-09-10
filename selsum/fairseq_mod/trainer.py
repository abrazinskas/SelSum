from fairseq.trainer import Trainer as BaseTrainer, logger, NanDetector
from fairseq.logging import meters, metrics
import torch
import contextlib
from fairseq import utils


class Trainer(BaseTrainer):
    """
    Adjusted valid_iterator to avoid problems with invalid validation instance
    sizes.
    """

    def get_valid_iterator(
        self,
        subset,
    ):
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=self.args.max_tokens_valid,
            max_sentences=self.args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                self.args.max_tokens,  # this one is added to avoid problems
            ),
            ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.args.num_workers
        )
