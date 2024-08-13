import os
import torch

from torchlogic.nn import LukasiewiczChannelAndBlock, LukasiewiczChannelOrBlock, LukasiewiczChannelXOrBlock, Predicates

from pytest import fixture

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(ROOT_DIR)


class TestLukasiewiczChannelBlock:

    @fixture
    def predicates(self):
        return Predicates(['feat1', 'feat2'])

    @staticmethod
    def test__forward_lukasiewicz_channel_block(predicates):
        block = LukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )

        # INPUT: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        x = torch.randn(32, 1, 10)
        out = block(x)
        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        assert out.size() == (32, 2, 1, 4)
        assert torch.all(out <= 1.0)
        assert torch.all(out >= 0.0)

        block = LukasiewiczChannelOrBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )

        # INPUT: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        x = torch.randn(32, 1, 10)
        out = block(x)
        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        assert out.size() == (32, 2, 1, 4)
        assert torch.all(out <= 1.0)
        assert torch.all(out >= 0.0)

        block = LukasiewiczChannelXOrBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )

        # INPUT: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        x = torch.randn(32, 1, 10)
        out = block(x)
        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        assert out.size() == (32, 2, 1, 4)
        assert torch.all(out <= 1.0)
        assert torch.all(out >= 0.0)