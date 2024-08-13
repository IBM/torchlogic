import os
from copy import deepcopy

import numpy as np
import torch

from torchlogic.nn import LukasiewiczChannelAndBlock, LukasiewiczChannelOrBlock, LukasiewiczChannelXOrBlock, \
    Predicates, ConcatenateBlocksLogic

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

        block3 = LukasiewiczChannelXOrBlock(
            channels=2,
            in_features=10,
            out_features=1,
            n_selected_features=5,
            parent_weights_dimension='out_features',
            operands=predicates,
            outputs_key='2'
        )

        cat_blocks = ConcatenateBlocksLogic([block1, block2, block3], '3')

        return cat_blocks

    @staticmethod
    def test__concatenated_blocks_produce_logic_string(concatenated_blocks):
        explanation = ["Predicate1", "Predicate2"]

        out = concatenated_blocks._produce_logic_string(concatenated_blocks.operands[0], explanation)
        assert out == "AND(Predicate1, Predicate2)", \
            "ConcatenateBlocksLogic produce_logic_string was not correct!"

        out = concatenated_blocks._produce_logic_string(concatenated_blocks.operands[1], explanation)
        assert out == "OR(Predicate1, Predicate2)", \
            "ConcatenateBlocksLogic produce_logic_string was not correct!"

        # out = concatenated_blocks._produce_logic_string(concatenated_blocks.operands[2], explanation)
        # assert out == "XOR(Predicate1, Predicate2)", \
        #     "ConcatenateBlocksLogic produce_logic_string was not correct!"

        out = concatenated_blocks._produce_logic_string(
            concatenated_blocks.operands[0], explanation, 'logical-natural')
        assert out == "ALL of the following are TRUE: \n\t- Predicate1, \n\t- Predicate2", \
            "ConcatenateBlocksLogic produce_logic_string was not correct!"

        out = concatenated_blocks._produce_logic_string(
            concatenated_blocks.operands[1], explanation, 'logical-natural')
        assert out == "ANY of the following are TRUE: \n\t- Predicate1, \n\t- Predicate2", \
            "ConcatenateBlocksLogic produce_logic_string was not correct!"

        # out = concatenated_blocks._produce_logic_string(
        #     concatenated_blocks.operands[2], explanation, 'logical-natural')
        # assert out == "ONE of the following is TRUE: \n\t- Predicate1, \n\t- Predicate2", \
        #     "ConcatenateBlocksLogic produce_logic_string was not correct!"

    @staticmethod
    def test__concatenated_blocks_produce_negation_string(concatenated_blocks):
        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], 1.0, "Predicate1", negate=False, explain_type='both')
        assert out == "Predicate1", "ConcatenateBlocksLogic produce_negation_string failed!"
        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], -1.0, "Predicate1", negate=False, explain_type='both')
        assert out == "NOT(Predicate1)", "ConcatenateBlocksLogic produce_negation_string failed!"

        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], 1.0, "Predicate1", negate=True, explain_type='both')
        assert out == "NOT(Predicate1)", "ConcatenateBlocksLogic produce_negation_string failed!"
        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], -1.0, "Predicate1", negate=True, explain_type='both')
        assert out == "Predicate1", "ConcatenateBlocksLogic produce_negation_string failed!"

        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], 1.0, "Predicate1", negate=False, explain_type='positive')
        assert out == "Predicate1", "ConcatenateBlocksLogic produce_negation_string failed!"
        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], -1.0, "Predicate1", negate=False, explain_type='positive')
        assert out == "", "ConcatenateBlocksLogic produce_negation_string failed!"

        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], 1.0, "Predicate1", negate=True, explain_type='positive')
        assert out == "", "ConcatenateBlocksLogic produce_negation_string failed!"
        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], -1.0, "Predicate1", negate=True, explain_type='positive')
        assert out == "Predicate1", "ConcatenateBlocksLogic produce_negation_string failed!"

        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], 1.0, "Predicate1", negate=False, explain_type='negative')
        assert out == "", "ConcatenateBlocksLogic produce_negation_string failed!"
        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], -1.0, "Predicate1", negate=False, explain_type='negative')
        assert out == "NOT(Predicate1)", "ConcatenateBlocksLogic produce_negation_string failed!"

        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], 1.0, "Predicate1", negate=True, explain_type='negative')
        assert out == "NOT(Predicate1)", "ConcatenateBlocksLogic produce_negation_string failed!"
        out = concatenated_blocks._produce_negation_string(
            concatenated_blocks.operands[0], -1.0, "Predicate1", negate=True, explain_type='negative')
        assert out == "", "ConcatenateBlocksLogic produce_negation_string failed!"

    @staticmethod
    def test__concatenated_blocks_produce_weights_mask(concatenated_blocks):
        out = concatenated_blocks._produce_weights_mask(torch.tensor([1.0, 0.5]), quantile=0.9)
        assert torch.all(out == torch.tensor([True, False])), "ConcatenateBlocksLogic produce_weights_mask failed!"
        out = concatenated_blocks._produce_weights_mask(torch.tensor([1.0, 0.5]), threshold=0.7)
        assert torch.all(out == torch.tensor([True, False])), "ConcatenateBlocksLogic produce_weights_mask failed!"

        out = concatenated_blocks._produce_weights_mask(torch.tensor([1.0, 1.0]), quantile=0.9)
        assert torch.all(out == torch.tensor([True, True])), "ConcatenateBlocksLogic produce_weights_mask failed!"
        out = concatenated_blocks._produce_weights_mask(torch.tensor([1.0, 1.0]), threshold=0.7)
        assert torch.all(out == torch.tensor([True, True])), "ConcatenateBlocksLogic produce_weights_mask failed!"

        out = concatenated_blocks._produce_weights_mask(torch.tensor([0.0, 0.0]), quantile=0.9)
        assert torch.all(out == torch.tensor([False, False])), "ConcatenateBlocksLogic produce_weights_mask failed!"
        out = concatenated_blocks._produce_weights_mask(torch.tensor([0.0, 0.0]), threshold=0.7)
        assert torch.all(out == torch.tensor([False, False])), "ConcatenateBlocksLogic produce_weights_mask failed!"

        out = concatenated_blocks._produce_weights_mask(torch.tensor([0.1, 0.1]), quantile=0.9)
        assert torch.all(out == torch.tensor([True, True])), "ConcatenateBlocksLogic produce_weights_mask failed!"
        out = concatenated_blocks._produce_weights_mask(torch.tensor([0.1, 0.1]), threshold=0.7)
        assert torch.all(out == torch.tensor([False, False])), "ConcatenateBlocksLogic produce_weights_mask failed!"

    @staticmethod
    def test__concatenated_blocks_explain(concatenated_blocks):
        concatenated_blocks.operands[0].weights.data.copy_(
            torch.tensor([[[0.5], [0.1], [1.0], [0.0], [5.0]],
                          [[0.4], [5.0], [0.2], [0.2], [1.0]]]))

        concatenated_blocks.operands[0].mask.data.copy_(
            torch.tensor([[[5, 4, 7, 3, 1]],
                          [[4, 9, 3, 8, 6]]]))

        concatenated_blocks.operands[1].weights.data.copy_(
            torch.tensor([[[0.00], [0.5], [0.0], [0.0], [0.75]],
                          [[0.4], [0.75], [0.2], [0.2], [1.0]]]))

        concatenated_blocks.operands[1].mask.data.copy_(
            torch.tensor([[[5, 4, 7, 3, 1]],
                          [[4, 9, 3, 8, 6]]]))

        CHANNELS = 2
        N_SELECTED_FEATURES = 3
        IN_FEATURES = 3
        OUT_FEATURES = 1

        out_block = LukasiewiczChannelOrBlock(
            channels=CHANNELS,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            n_selected_features=N_SELECTED_FEATURES,
            parent_weights_dimension='out_features',
            operands=concatenated_blocks,
            outputs_key='out_block'
        )

        # begin tests of different configurations

        # test 1: only channel 0, 50%
        out_block.weights.data.copy_(torch.tensor([[[1.0], [0.0], [0.0]], [[0.5], [0.0], [0.0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain(
            quantile=0.4,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == ['AND(feat1 >= 0.985, feat7 >= 0.925)'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        # test 2: only channel 1, 50%.
        out_block.weights.data.copy_(torch.tensor([[[0.5], [0], [0]], [[1.0], [0], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))  # out_feature index
        out = out_block.explain(
            quantile=0.4,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=1
        )
        assert out == ['AND(feat6 >= 0.94, feat9 >= 0.988)'], "ConcatenateBlocksLogic did not explain correctly!"

        # test 3: channel 0, 100%.
        out_block.weights.data.copy_(torch.tensor([[[1.0], [0], [0]], [[1.0], [0], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain(
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == ['AND(feat1 >= 0.985, feat4 >= 0.25, feat5 >= 0.85, feat7 >= 0.925)'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output
        out_block.weights.data.copy_(torch.tensor([[[0], [1.0], [0]], [[0], [1.0], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain(
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == ['OR(feat1 >= 0.467, feat4 >= 0.699)'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. not strictly less than. negation
        out_block.weights.data.copy_(torch.tensor([[[0], [-1.0], [0]], [[0], [0.5], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain(
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == [''], \
            "ConcatenateBlocksLogic did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. one input below threshold. negation
        out_block.weights.data.copy_(torch.tensor([[[0], [-0.9], [0]], [[0], [0.5], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain(
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == ['NOT(OR(feat1 >= 0.148, feat4 >= 0.222))'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. one input below threshold. negation. negation on pred
        concatenated_blocks.operands[1].weights.data.copy_(
            torch.tensor([[[0.5], [0.1], [1.0], [0.0], [-5.0]],
                          [[0.4], [5.0], [0.2], [0.2], [1.0]]]))

        out_block.weights.data.copy_(torch.tensor([[[0], [-1.0], [0]], [[0], [0.5], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain(
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == ['NOT(OR(NOT(feat1 >= 0.985), feat4 >= 0.75, feat5 >= 0.15, feat7 >= 0.075))'], \
            "ConcatenateBlocksLogic did not explain correctly!"

    @staticmethod
    def test__concatenated_blocks_explain_sample(concatenated_blocks):
        concatenated_blocks.operands[0].weights.data.copy_(
            torch.tensor([[[0.5], [0.1], [1.0], [0.0], [5.0]],
                          [[0.4], [5.0], [0.2], [0.2], [1.0]]]))

        concatenated_blocks.operands[0].mask.data.copy_(
            torch.tensor([[[5, 4, 7, 3, 1]],
                          [[4, 9, 3, 8, 6]]]))

        concatenated_blocks.operands[1].weights.data.copy_(
            torch.tensor([[[0.00], [0.6], [0.0], [0.0], [0.75]],
                          [[0.4], [0.75], [0.2], [0.2], [1.0]]]))

        concatenated_blocks.operands[1].mask.data.copy_(
            torch.tensor([[[5, 4, 7, 3, 1]],
                          [[4, 9, 3, 8, 6]]]))

        CHANNELS = 2
        N_SELECTED_FEATURES = 3
        IN_FEATURES = 3
        OUT_FEATURES = 1

        out_block = LukasiewiczChannelOrBlock(
            channels=CHANNELS,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            n_selected_features=N_SELECTED_FEATURES,
            parent_weights_dimension='out_features',
            operands=concatenated_blocks,
            outputs_key='out_block'
        )

        # begin tests of different configurations

        # test 1: only channel 0, 40%
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[1.]], [[1.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[1., 1., 1.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[1.]], [[1.]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[1.0], [0], [0]], [[0.5], [0], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=deepcopy(outputs_dict),
            quantile=0.4,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['AND(feat1 >= 0.94, feat7 >= 0.7)'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        out = out_block.explain_sample(
            outputs_dict=deepcopy(outputs_dict),
            quantile=0.4,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['feat1 >= 1.0'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        out = out_block.explain_sample(
            outputs_dict=deepcopy(outputs_dict),
            quantile=0.4,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['feat7 >= 1.0'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        out = out_block.explain_sample(
            outputs_dict=deepcopy(outputs_dict),
            quantile=0.4,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0]),
            channel=0
        )
        assert out == ['AND(feat1 >= 0.96, feat7 >= 0.7)'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        # test 2: only channel 0, 40%, output is below threshold
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[1.]], [[1.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[1., 1., 1.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[0.0]], [[1.]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[1.0], [0], [0]], [[0.5], [0], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=deepcopy(outputs_dict),
            quantile=0.4,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == [''], "ConcatenateBlocksLogic did not explain correctly!"

        # test 2: only channel 1, 40%.
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[1.]], [[1.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[1., 1., 1.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[1.]], [[1.]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[0], [0.5], [0]], [[0], [1.0], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=0.4,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=1
        )
        assert out == ['OR(feat6 >= 0.0, feat9 >= 0.0)'], "ConcatenateBlocksLogic did not explain correctly!"

        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=0.4,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=1
        )
        assert out == ['OR(feat6 >= 0.0, feat9 >= 0.0)'], "ConcatenateBlocksLogic did not explain correctly!"

        # test 2: only channel 1, 40%, output is below threshold
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[1.]], [[1.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[1., 1., 1.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[1.]], [[0.0]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[0], [0.5], [0]], [[0], [1.0], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=0.5,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=1
        )
        assert out == [''], "ConcatenateBlocksLogic did not explain correctly!"

        # test 3: channel 0, 100%.
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[1.]], [[1.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[1., 1., 1.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[1.]], [[1.]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[1.0], [0], [0]], [[0.5], [0], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['AND(feat1 >= 0.94, feat4 >= 0.0, feat5 >= 0.4, feat7 >= 0.7)'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        # test 3: channel 0 100% output below threshold
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[1.]], [[1.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[1., 1., 1.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[0.0]], [[1.]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[1.0], [0], [0]], [[0.5], [0], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == [''], "ConcatenateBlocksLogic did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[1.]], [[1.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[1., 1., 1.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[1.]], [[1.]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[0], [1.0], [0]], [[0], [0.5], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['OR(feat1 >= 0.133, feat4 >= 0.042)'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['OR(feat1 >= 0.133, feat4 >= 0.0)'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. input below threshold
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[1.]], [[1.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[1., 1., 1.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[0.]], [[1.]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[0], [1.0], [0]], [[0], [0.5], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == [''], "ConcatenateBlocksLogic did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. one input below threshold
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[1.]], [[1.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[1., 1., 1.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[1.]], [[1.]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[0], [1.0], [0]], [[0], [0.5], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['feat4 >= 1.0'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. one input below threshold. negation
        concatenated_blocks.operands[1].weights.data.copy_(
            torch.tensor([[[0.0], [0.5], [0.0], [0.0], [0.75]],
                          [[0.4], [5.0], [0.2], [0.2], [1.0]]]))
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[0.]], [[0.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[1., 0., 1.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[1.]], [[1.]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[0], [-1.0], [0]], [[0], [0.5], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 0.199, 1.0, 1.0, 0.299, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0,
            print_type='logical'
        )
        assert out == ['NOT(OR(feat1 >= 0.201, feat4 >= 0.302))'], \
            "ConcatenateBlocksLogic did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. one input below threshold. negation. negation on pred
        concatenated_blocks.operands[1].weights.data.copy_(
            torch.tensor([[[0.0], [0.5], [0.0], [0.0], [-0.75]],
                          [[0.4], [5.0], [0.2], [0.2], [1.0]]]))
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            '0': torch.tensor([[[[1.]], [[1.]]]]),
            '1': torch.tensor([[[[0.]], [[0.]]]]),
            '2': torch.tensor([[[[1.]], [[1.]]]]),
            '3': torch.tensor([[[[0., 0., 0.]], [[1., 1., 1.]]]]),
            'out_block': torch.tensor([[[[1.]], [[1.]]]])
        }
        out_block.weights.data.copy_(torch.tensor([[[0], [-1.0], [0]], [[0], [0.5], [0]]]))
        out_block.mask.data.copy_(torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))
        out = out_block.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.ones(1),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 0.801, 1.0, 1.0, 0.299, 1.0, 1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['NOT(OR(NOT(feat1 >= 0.799), feat4 >= 0.302))'], \
            "ConcatenateBlocksLogic did not explain correctly!"
