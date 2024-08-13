import os
from copy import deepcopy

import torch

from torchlogic.nn import (LukasiewiczChannelAndBlock, LukasiewiczChannelOrBlock, LukasiewiczChannelXOrBlock,
                           Predicates, VariationalLukasiewiczChannelAndBlock, VariationalLukasiewiczChannelOrBlock,
                           AttentionLukasiewiczChannelAndBlock, AttentionLukasiewiczChannelOrBlock)

from pytest import fixture

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(ROOT_DIR)


class TestLukasiewiczChannelBlock:

    @fixture
    def predicates(self):
        return Predicates(['feat1', 'feat2'])

    @staticmethod
    def test__init_lukasiewicz_channel_block(predicates):
        block = LukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )

        assert block.weights.size() == (2, 5, 4), "LukasiewiczChannelBlock `weights` did not initialize correctly!"
        assert block.mask.size() == (2, 4, 5), "LukasiewiczChannelBlock `mask` did not initialize correctly!"
        assert block.logic_type == 'And', "LukasiewiczChannelAndBlock `logic_type` did not initialize correctly!"

        block = LukasiewiczChannelOrBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )

        assert block.logic_type == 'Or', "LukasiewiczChannelOrBlock `logic_type` did not initialize correctly!"

        block = LukasiewiczChannelXOrBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )

        assert block.logic_type == 'XOr', "LukasiewiczChannelXOrBlock `logic_type` did not initialize correctly!"

    @staticmethod
    def test__lukasiewicz_channel_block_produce_logic_string(predicates):
        block = LukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )
        explanation = ["Predicate1", "Predicate2"]
        out = block._produce_logic_string(explanation)
        assert out == "AND(Predicate1, Predicate2)", \
            "LukasiewiczChannelAndBlock produce_logic_string was not correct!"

        block = LukasiewiczChannelOrBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )
        explanation = ["Predicate1", "Predicate2"]
        out = block._produce_logic_string(explanation)
        assert out == "OR(Predicate1, Predicate2)", \
            "LukasiewiczChannelOrBlock produce_logic_string was not correct!"

        block = LukasiewiczChannelXOrBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )
        explanation = ["Predicate1", "Predicate2"]
        out = block._produce_logic_string(explanation)
        assert out == "XOR(Predicate1, Predicate2)", \
            "LukasiewiczChannelXOrBlock produce_logic_string was not correct!"

        block = LukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )
        explanation = ["Predicate1", "Predicate2"]
        out = block._produce_logic_string(explanation, 'logical-natural')
        assert out == "ALL the following are TRUE: \n\t- Predicate1, \n\t- Predicate2", \
            "LukasiewiczChannelAndBlock produce_logic_string was not correct!"

        block = LukasiewiczChannelOrBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )
        explanation = ["Predicate1", "Predicate2"]
        out = block._produce_logic_string(explanation, 'logical-natural')
        assert out == "ANY of the following are TRUE: \n\t- Predicate1, \n\t- Predicate2", \
            "LukasiewiczChannelOrBlock produce_logic_string was not correct!"

        block = LukasiewiczChannelXOrBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )
        explanation = ["Predicate1", "Predicate2"]
        out = block._produce_logic_string(explanation, 'logical-natural')
        assert out == "ONE of the following is TRUE: \n\t- Predicate1, \n\t- Predicate2", \
            "LukasiewiczChannelXOrBlock produce_logic_string was not correct!"

    @staticmethod
    def test__lukasiewicz_channel_block_produce_negation_string(predicates):
        block = LukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )
        out = block._produce_negation_string(1.0, "Predicate1", negate=False, explain_type='both')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(-1.0, "Predicate1", negate=False, explain_type='both')
        assert out == "NOT(Predicate1)", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(1.0, "Predicate1", negate=True, explain_type='both')
        assert out == "NOT(Predicate1)", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(-1.0, "Predicate1", negate=True, explain_type='both')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(1.0, "Predicate1", negate=False, explain_type='positive')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(-1.0, "Predicate1", negate=False, explain_type='positive')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(1.0, "Predicate1", negate=True, explain_type='positive')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(-1.0, "Predicate1", negate=True, explain_type='positive')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(1.0, "Predicate1", negate=False, explain_type='negative')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(-1.0, "Predicate1", negate=False, explain_type='negative')
        assert out == "NOT(Predicate1)", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(1.0, "Predicate1", negate=True, explain_type='negative')
        assert out == "NOT(Predicate1)", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(-1.0, "Predicate1", negate=True, explain_type='negative')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"

        # logical-natural
        out = block._produce_negation_string(
            1.0, "Predicate1", negate=False, explain_type='both', print_type='logical-natural')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=False, explain_type='both', print_type='logical-natural')
        assert out == 'NOT the following: \n\t- Predicate1', "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(
            1.0, "Predicate1", negate=True, explain_type='both', print_type='logical-natural')
        assert out == 'NOT the following: \n\t- Predicate1', "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=True, explain_type='both', print_type='logical-natural')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(
            1.0, "Predicate1", negate=False, explain_type='positive', print_type='logical-natural')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=False, explain_type='positive', print_type='logical-natural')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(
            1.0, "Predicate1", negate=True, explain_type='positive', print_type='logical-natural')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=True, explain_type='positive', print_type='logical-natural')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(
            1.0, "Predicate1", negate=False, explain_type='negative', print_type='logical-natural')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=False, explain_type='negative', print_type='logical-natural')
        assert out == 'NOT the following: \n\t- Predicate1', "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(
            1.0, "Predicate1", negate=True, explain_type='negative', print_type='logical-natural')
        assert out == 'NOT the following: \n\t- Predicate1', "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=True, explain_type='negative', print_type='logical-natural')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"

        # natural
        out = block._produce_negation_string(
            1.0, "Predicate1", negate=False, explain_type='both', print_type='natural')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=False, explain_type='both', print_type='natural')
        assert out == 'it was NOT true Predicate1', "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(
            1.0, "Predicate1", negate=True, explain_type='both', print_type='natural')
        assert out == 'it was NOT true Predicate1', "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=True, explain_type='both', print_type='natural')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(
            1.0, "Predicate1", negate=False, explain_type='positive', print_type='natural')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=False, explain_type='positive', print_type='natural')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(
            1.0, "Predicate1", negate=True, explain_type='positive', print_type='natural')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=True, explain_type='positive', print_type='natural')
        assert out == "Predicate1", "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(
            1.0, "Predicate1", negate=False, explain_type='negative', print_type='natural')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=False, explain_type='negative', print_type='natural')
        assert out == 'it was NOT true Predicate1', "LukasiewiczChannelBlock produce_negation_string failed!"

        out = block._produce_negation_string(
            1.0, "Predicate1", negate=True, explain_type='negative', print_type='natural')
        assert out == 'it was NOT true Predicate1', "LukasiewiczChannelBlock produce_negation_string failed!"
        out = block._produce_negation_string(
            -1.0, "Predicate1", negate=True, explain_type='negative', print_type='natural')
        assert out == "", "LukasiewiczChannelBlock produce_negation_string failed!"

    @staticmethod
    def test__lukasiewicz_channel_block_produce_weights_mask(predicates):
        block = LukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )
        out = block._produce_weights_mask(torch.tensor([1.0, 0.5]), quantile=0.9)
        assert torch.all(out == torch.tensor([True, False])), "LukasiewiczChannelBlock produce_weights_mask failed!"
        out = block._produce_weights_mask(torch.tensor([1.0, 0.5]), threshold=0.7)
        assert torch.all(out == torch.tensor([True, False])), "LukasiewiczChannelBlock produce_weights_mask failed!"

        out = block._produce_weights_mask(torch.tensor([1.0, 1.0]), quantile=0.9)
        assert torch.all(out == torch.tensor([True, True])), "LukasiewiczChannelBlock produce_weights_mask failed!"
        out = block._produce_weights_mask(torch.tensor([1.0, 1.0]), threshold=0.7)
        assert torch.all(out == torch.tensor([True, True])), "LukasiewiczChannelBlock produce_weights_mask failed!"

        out = block._produce_weights_mask(torch.tensor([0.0, 0.0]), quantile=0.9)
        assert torch.all(out == torch.tensor([False, False])), "LukasiewiczChannelBlock produce_weights_mask failed!"
        out = block._produce_weights_mask(torch.tensor([0.0, 0.0]), threshold=0.7)
        assert torch.all(out == torch.tensor([False, False])), "LukasiewiczChannelBlock produce_weights_mask failed!"

        out = block._produce_weights_mask(torch.tensor([0.1, 0.1]), quantile=0.9)
        assert torch.all(out == torch.tensor([True, True])), "LukasiewiczChannelBlock produce_weights_mask failed!"
        out = block._produce_weights_mask(torch.tensor([0.1, 0.1]), threshold=0.7)
        assert torch.all(out == torch.tensor([False, False])), "LukasiewiczChannelBlock produce_weights_mask failed!"

    @staticmethod
    def test__lukasiewicz_channel_block_explain():
        predicates = Predicates(feature_names=[f"Predicate{i}" for i in range(4)])

        block1 = LukasiewiczChannelAndBlock(
            channels=2,
            in_features=4,
            out_features=2,
            n_selected_features=2,
            parent_weights_dimension='out_features',
            operands=predicates,
            outputs_key='0'
        )

        # 1.0 vs 0.2 -> 0 vs 3
        # 5.0 vs 0.3 -> 1 vs 0
        # 0.1 vs 5.0 -> 1 vs 2
        # 0.7 vs 1.0 -> 2 vs 3
        block1.weights.data.copy_(torch.tensor([[[1.0, 5.0], [0.2, 0.3]],
                                                [[0.1, 0.7], [5.0, 1.0]]]))

        # left side is channel 0, right is channel 1
        block1.mask.data.copy_(torch.tensor([[[0, 3], [1, 0]],
                                             [[1, 2], [2, 3]]]))

        CHANNELS = 2
        N_SELECTED_FEATURES = 1
        IN_FEATURES = 2
        OUT_FEATURES = 1

        block2 = LukasiewiczChannelOrBlock(
            channels=CHANNELS,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            n_selected_features=N_SELECTED_FEATURES,
            parent_weights_dimension='out_features',
            operands=block1,
            outputs_key='0'
        )

        # begin tests of different configurations

        # test 1: only channel 0, 50%
        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[0]], [[0]]]))
        out = block2.explain(
            quantile=0.5,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == ['Predicate0 >= 0.85'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[0]], [[0]]]))
        out = block2.explain(
            quantile=0.5,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0,
            print_type='logical-natural'
        )
        assert out == ['Predicate0 greater than or equal to 0.85'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[0]], [[0]]]))
        out = block2.explain(
            quantile=0.5,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0,
            print_type='natural'
        )
        assert out == ['Predicate0 greater than or equal to 0.85'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        # test 2: only channel 1, 50%.
        block2.weights.data.copy_(torch.tensor([[[0.5]], [[1.0]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain(
            quantile=0.5,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=1
        )
        assert out == ['Predicate3 >= 0.85'], "LukasiewiczChannelBlock did not explain correctly!"

        # test 3: channel 0, 100%.
        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[0]], [[0]]]))
        out = block2.explain(
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == ['AND(Predicate0 >= 0.85, Predicate3 >= 0.25)'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[0]], [[0]]]))
        out = block2.explain(
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0,
            print_type='logical-natural'
        )
        assert out == ['ALL the following are TRUE: \n\t- Predicate0 greater than or equal to 0.85, '
                       '\n\t- Predicate3 greater than or equal to 0.25'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[0]], [[0]]]))
        out = block2.explain(
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0,
            print_type='natural'
        )
        assert out == ['Predicate0 greater than or equal to 0.85, and Predicate3 greater than or equal to 0.25'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output
        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain(
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == ['AND(Predicate0 >= 0.5, Predicate1 >= 0.97)'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. one input below threshold. negation
        block2.weights.data.copy_(torch.tensor([[[-1.0]], [[0.0]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain(
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == ['NOT(Predicate1 >= 0.92)'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        block2.weights.data.copy_(torch.tensor([[[-1.0]], [[0.0]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain(
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0,
            print_type='logical-natural'
        )
        assert out == ['NOT the following: '
                       '\n\t- Predicate1 greater than or equal to 0.92'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        block2.weights.data.copy_(torch.tensor([[[-1.0]], [[0.0]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain(
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0,
            print_type='natural'
        )
        assert out == ['it was NOT true Predicate1 greater than or equal to 0.92'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. one input below threshold. negation. negation on pred
        block1.weights.data.copy_(torch.tensor([[[1.0, -5.0], [0.2, 0.3]],
                                                [[0.1, 0.7], [5.0, 1.0]]]))
        block2.weights.data.copy_(torch.tensor([[[-1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain(
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0
        )
        assert out == ['NOT(NOT(Predicate1 >= 0.08))'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        block1.weights.data.copy_(torch.tensor([[[1.0, -5.0], [0.2, 0.3]],
                                                [[0.1, 0.7], [5.0, 1.0]]]))
        block2.weights.data.copy_(torch.tensor([[[-1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain(
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0,
            print_type='logical-natural'
        )
        assert out == ['NOT the following: '
                       '\n\t- NOT Predicate1 greater than or equal to 0.08'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        block1.weights.data.copy_(torch.tensor([[[1.0, -5.0], [0.2, 0.3]],
                                                [[0.1, 0.7], [5.0, 1.0]]]))
        block2.weights.data.copy_(torch.tensor([[[-1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain(
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            channel=0,
            print_type='natural'
        )
        assert out == ['Predicate1 greater than or equal to 0.08'], \
            "LukasiewiczChannelBlock did not explain correctly!"

    @staticmethod
    def test__lukasiewicz_channel_block_explain_sample():
        predicates = Predicates(feature_names=[f"Predicate{i}" for i in range(4)])

        block1 = LukasiewiczChannelAndBlock(
            channels=2,
            in_features=4,
            out_features=2,
            n_selected_features=2,
            parent_weights_dimension='out_features',
            operands=predicates,
            outputs_key='block1'
        )

        # 1.0 vs 0.2 -> 0 vs 3
        # 5.0 vs 0.3 -> 1 vs 0
        # 0.1 vs 5.0 -> 1 vs 2
        # 0.7 vs 1.0 -> 2 vs 3
        block1.weights.data.copy_(torch.tensor([[[1.0, 5.0], [0.2, 0.3]],
                                                [[0.1, 0.7], [5.0, 1.0]]]))

        # left side is channel 0, right is channel 1
        block1.mask.data.copy_(torch.tensor([[[0, 3], [1, 0]],
                                             [[1, 2], [2, 3]]]))

        CHANNELS = 2
        N_SELECTED_FEATURES = 1
        IN_FEATURES = 2
        OUT_FEATURES = 1

        block2 = LukasiewiczChannelOrBlock(
            channels=CHANNELS,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            n_selected_features=N_SELECTED_FEATURES,
            parent_weights_dimension='out_features',
            operands=block1,
            outputs_key='block2'
        )

        # begin tests of different configurations

        # test 1: only channel 0, 50%
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[1., 1.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[1.]], [[1.]]]])
        }
        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[0]], [[0]]]))
        out = block2.explain_sample(
            outputs_dict=deepcopy(outputs_dict),
            quantile=0.5,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['Predicate0 >= 0.7'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        # test 2: only channel 0, 50%, output is below threshold
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[1., 1.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[0.0]], [[1.0]]]])
        }
        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[0]], [[0]]]))
        out = block2.explain_sample(
            outputs_dict=deepcopy(outputs_dict),
            quantile=0.5,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == [''], "LukasiewiczChannelBlock did not explain correctly!"

        # test 2: only channel 1, 50%.
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[1., 1.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[1.0]], [[1.0]]]])
        }
        block2.weights.data.copy_(torch.tensor([[[0.5]], [[1.0]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain_sample(
            outputs_dict=outputs_dict,
            quantile=0.5,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            channel=1
        )
        assert out == ['Predicate3 >= 0.7'], "LukasiewiczChannelBlock did not explain correctly!"

        # test 2: only channel 1, 50%, output is below threshold
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[1., 1.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[1.0]], [[.0]]]])
        }
        block2.weights.data.copy_(torch.tensor([[[0.5]], [[1.0]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain_sample(
            outputs_dict=outputs_dict,
            quantile=0.5,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            channel=1
        )
        assert out == [''], "LukasiewiczChannelBlock did not explain correctly!"

        # test 3: channel 0, 100%.
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[1., 1.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[1.0]], [[1.0]]]])
        }
        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[0]], [[0]]]))
        out = block2.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['AND(Predicate0 >= 0.7, Predicate3 >= 0.0)'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        # test 3: channel 0 100% output below threshold
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[1., 1.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[0.0]], [[1.0]]]])
        }
        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[0]], [[0]]]))
        out = block2.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == [''], "LukasiewiczChannelBlock did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[1., 1.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[1.0]], [[0.0]]]])
        }
        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['AND(Predicate0 >= 0.0, Predicate1 >= 0.94)'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. input below threshold
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[1., 1.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[1.0]], [[0.0]]]])
        }
        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([0.0, 0.0, 1.0, 1.0]),
            channel=0
        )
        assert out == [''], "LukasiewiczChannelBlock did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. one input below threshold
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[1., 1.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[1.0]], [[0.0]]]])
        }
        block2.weights.data.copy_(torch.tensor([[[1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([0.5, 1.0, 1.0, 1.0]),
            channel=0
        )
        assert out == ['AND(Predicate0 >= 0.0, Predicate1 >= 0.97)'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. one input below threshold. negation
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[0., 0.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[1.0]], [[0.5]]]])
        }
        block2.weights.data.copy_(torch.tensor([[[-1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([0.499, 0.889, 1.0, 1.0]),
            channel=0
        )
        assert out == ['NOT(AND(Predicate0 >= 0.517, Predicate1 >= 0.89))'], \
            "LukasiewiczChannelBlock did not explain correctly!"

        # test 4: only channel 0, 100%. weights from other output. one input below threshold. negation. negation on pred
        outputs_dict = {
            # 1, CHANNELS, 1, OUT FEATURES
            'block1': torch.tensor([[[[0., 0.]], [[1., 1.]]]]),
            'block2': torch.tensor([[[[1.0]], [[0.5]]]])
        }
        block1.weights.data.copy_(torch.tensor([[[1.0, -5.0], [0.2, 0.3]],
                                                [[0.1, 0.7], [5.0, 1.0]]]))
        block2.weights.data.copy_(torch.tensor([[[-1.0]], [[0.5]]]))
        block2.mask.data.copy_(torch.tensor([[[1]], [[1]]]))
        out = block2.explain_sample(
            outputs_dict=outputs_dict,
            quantile=1.0,
            parent_weights=torch.tensor(1.),
            parent_mask=torch.tensor([0]),
            explain_type='both',
            required_output_thresholds=torch.tensor(0.7),
            parent_logic_type='And',
            input_features=torch.tensor([0.499, 0.111, 1.0, 1.0]),
            channel=0
        )
        assert out == ['NOT(AND(NOT(Predicate1 >= 0.11), Predicate0 >= 0.517))'], \
            "LukasiewiczChannelBlock did not explain correctly!"

    @staticmethod
    def test__init_attention_lukasiewicz_channel_block(predicates):
        block = AttentionLukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0',
            attn_emb_dim=32,
            attn_n_layers=3
        )

        assert block.weights.size() == (2, 5, 4), "LukasiewiczChannelBlock `weights` did not initialize correctly!"
        assert block.mask.size() == (2, 4, 5), "LukasiewiczChannelBlock `mask` did not initialize correctly!"
        assert block.logic_type == 'And', "LukasiewiczChannelAndBlock `logic_type` did not initialize correctly!"
        # subtract 1 because there should be 1 relu activations, 1 EXU layer and 2 linear layers
        assert len(block.attn) - 1 == 3, ("AttentionLukasiewiczChannelAndBlock `attn`"
                                          " network has wrong number of layers")
        # dimension 1 of all weights should be var_emb_dim sized
        for l in block.attn:
            if isinstance(l, torch.nn.Linear):
                assert l.weight.size(1) == 16 or l.weight.size(1) == 32, \
                    "attn does not have correct hidden size or output size"
                assert l.weight.size(0) == 16 or l.weight.size(0) == 1, \
                    "attn does not have correct hidden size or input size"

        assert block.tau.size() == (1, 2, 4, 1, 1), "tau does not have correct size"

        block = AttentionLukasiewiczChannelOrBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )

        assert block.logic_type == 'Or', "LukasiewiczChannelOrBlock `logic_type` did not initialize correctly!"

    @staticmethod
    def test__sample_mask_attention_lukasiewicz_channel_block(predicates):
        block = AttentionLukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0',
            attn_emb_dim=32,
            attn_n_layers=3
        )

        # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        size = (32, 2, 4, 1, 5)
        x = torch.randn(size=size)
        mask = block.sample_mask(x)
        assert mask.size() == size, "sample_mask did not return tensor of correct size"

    @staticmethod
    def test__produce_explanation_weights_attention_lukasiewicz_channel_block(predicates):
        block = AttentionLukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0',
            attn_emb_dim=32,
            attn_n_layers=3
        )

        # X: BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        in_size = (32, 2, 1, 10)
        # OUT: BATCH_SIZE, CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        out_size = (32, 2, 5, 4)
        x = torch.randn(size=in_size)
        mask = block.produce_explanation_weights(x)
        assert mask.size() == out_size, "produce_explanation_weights did not return tensor of correct size"

    @staticmethod
    def test__init_variational_lukasiewicz_channel_block(predicates):
        block = VariationalLukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0',
            var_emb_dim=32,
            var_n_layers=3
        )

        assert block.weights.size() == (2, 5, 4), \
            "VariationalLukasiewiczChannelBlock `weights` did not initialize correctly!"
        assert block.mask.size() == (2, 4, 5), \
            "VariationalLukasiewiczChannelBlock `mask` did not initialize correctly!"
        assert block.logic_type == 'And', \
            "VariationalLukasiewiczChannelAndBlock `logic_type` did not initialize correctly!"
        # subtract 2 because there should be 2 relu activations and 3 linear layers
        assert len(block.var_mean) - 2 == 3, ("VariationalLukasiewiczChannelAndBlock `var_mean`"
                                              " network has wrong number of layers")
        # subtract 2 because there should be 2 relu activations and 3 linear layers
        assert len(block.var_std) - 2 == 3, ("VariationalLukasiewiczChannelAndBlock `var_mean`"
                                             " network has wrong number of layers")
        # dimension 1 of all weights should be var_emb_dim sized
        for l in block.var_mean:
            if isinstance(l, torch.nn.Linear):
                assert l.weight.size(1) == 32, "var_mean does not have correct hidden size"
        # dimension 1 of all weights should be var_emb_dim sized
        for l in block.var_std:
            if isinstance(l, torch.nn.Linear):
                assert l.weight.size(1) == 32, "var_std does not have correct hidden size"
        assert block.var_mean[0].weight.size(0) == 32, "var_mean does not have correct input size"
        assert block.var_std[0].weight.size(0) == 32, "var_std does not have correct input size"
        assert block.var_mean[-1].weight.size(0) == block.weights.size(1), "var_mean does not have correct output size"
        assert block.var_std[-1].weight.size(0) == block.weights.size(1), "var_std does not have correct output size"
        assert block.tau.size() == (2, 1, 4), "tau does not have correct size"

        block = VariationalLukasiewiczChannelOrBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0'
        )

        assert block.logic_type == 'Or', "LukasiewiczChannelOrBlock `logic_type` did not initialize correctly!"

    @staticmethod
    def test__sample_mask_variational_lukasiewicz_channel_block(predicates):
        block = VariationalLukasiewiczChannelAndBlock(
            channels=2,
            in_features=10,
            out_features=4,
            n_selected_features=5,
            parent_weights_dimension='channels',
            operands=predicates,
            outputs_key='0',
            var_emb_dim=32,
            var_n_layers=3
        )

        # WEIGHTS MASK: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        size = (2, 5, 4)
        mask = block.sample_mask()
        assert mask.size() == size, "sample_mask did not return tensor of correct size"
