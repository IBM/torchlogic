import os
import torch

from torchlogic.nn import LukasiewiczChannelAndBlock, LukasiewiczChannelOrBlock, Predicates, ConcatenateBlocksLogic

from pytest import fixture

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(ROOT_DIR)


class TestLukasiewiczChannelBlock:

    @fixture
    def predicates(self):
        return Predicates([f'feat{i}' for i in range(10)])

    @fixture
    def concatenated_blocks(self, predicates):
        block1 = LukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=1,
            n_selected_features=5,
            parent_weights_dimension='out_features',
            operands=predicates,
            outputs_key='0'
        )

        block2 = LukasiewiczChannelOrBlock(
            channels=2,
            in_features=10,
            out_features=1,
            n_selected_features=5,
            parent_weights_dimension='out_features',
            operands=predicates,
            outputs_key='1'
        )

        cat_blocks = ConcatenateBlocksLogic([block1, block2], '2')

        return cat_blocks

    @staticmethod
    def test__forward_lukasiewicz_channel_block(concatenated_blocks):
        # INPUT: [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        x1 = torch.rand(32, 2, 1, 1)
        x2 = torch.rand(32, 2, 1, 1)
        out = concatenated_blocks(x1, x2)
        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        assert out.size() == (32, 2, 1, 2)
        assert torch.all(out <= 1.0)
        assert torch.all(out >= 0.0)
