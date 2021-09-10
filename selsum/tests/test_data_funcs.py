import unittest
from selsum.utils.helpers.model import create_blocker_mask
import torch as T


class TestDataFuncs(unittest.TestCase):

    def test_blocker_mask(self):
        """"""
        # 4 is padding
        prev_sel_indxs = T.tensor([[4, 1, 2, 0], [4, 0, 3, 1], [4, 3, 2, 1]],
                                  dtype=T.long)
        expected_bmask = T.tensor([[
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 1, 1, 0, 1],
            [1, 1, 1, 0, 1]
        ], [
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1]
        ], [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1]
        ]], dtype=T.bool)
        blocker_mask = create_blocker_mask(prev_sel_indxs)
        self.assertTrue((expected_bmask == blocker_mask).all())


if __name__ == '__main__':
    unittest.main()
