import os

import torch

from torchlogic.nn import Predicates

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(ROOT_DIR)


class TestPredicates:

    @staticmethod
    def test__init_predicates():
        predicates = Predicates(feature_names=[f"Predicate{i}" for i in range(10)])
        assert predicates.operands == [f"Predicate{i}" for i in range(10)]

    @staticmethod
    def test__predicates_produce_negation_string():
        predicates = Predicates(feature_names=[f"Predicate{i}" for i in range(10)])

        out = predicates._produce_negation_string(1.0, "Predicate1", negate=False, explain_type='both')
        assert out == "Predicate1", "LukasiewiczLayer produce_negation_string failed!"
        out = predicates._produce_negation_string(-1.0, "Predicate1", negate=False, explain_type='both')
        assert out == "NOT(Predicate1)", "LukasiewiczLayer produce_negation_string failed!"

        out = predicates._produce_negation_string(1.0, "Predicate1", negate=True, explain_type='both')
        assert out == "NOT(Predicate1)", "LukasiewiczLayer produce_negation_string failed!"
        out = predicates._produce_negation_string(-1.0, "Predicate1", negate=True, explain_type='both')
        assert out == "Predicate1", "LukasiewiczLayer produce_negation_string failed!"

        out = predicates._produce_negation_string(1.0, "Predicate1", negate=False, explain_type='positive')
        assert out == "Predicate1", "LukasiewiczLayer produce_negation_string failed!"
        out = predicates._produce_negation_string(-1.0, "Predicate1", negate=False, explain_type='positive')
        assert out == "", "LukasiewiczLayer produce_negation_string failed!"

        out = predicates._produce_negation_string(1.0, "Predicate1", negate=True, explain_type='positive')
        assert out == "", "LukasiewiczLayer produce_negation_string failed!"
        out = predicates._produce_negation_string(-1.0, "Predicate1", negate=True, explain_type='positive')
        assert out == "Predicate1", "LukasiewiczLayer produce_negation_string failed!"

        out = predicates._produce_negation_string(1.0, "Predicate1", negate=False, explain_type='negative')
        assert out == "", "LukasiewiczLayer produce_negation_string failed!"
        out = predicates._produce_negation_string(-1.0, "Predicate1", negate=False, explain_type='negative')
        assert out == "NOT(Predicate1)", "LukasiewiczLayer produce_negation_string failed!"

        out = predicates._produce_negation_string(1.0, "Predicate1", negate=True, explain_type='negative')
        assert out == "NOT(Predicate1)", "LukasiewiczLayer produce_negation_string failed!"
        out = predicates._produce_negation_string(-1.0, "Predicate1", negate=True, explain_type='negative')
        assert out == "", "LukasiewiczLayer produce_negation_string failed!"

    @staticmethod
    def test__predicates_produce_weights_mask():
        predicates = Predicates(feature_names=[f"Predicate{i}" for i in range(10)])

        out = predicates._produce_weights_mask(torch.tensor([1.0, 0.5]), quantile=0.9)
        assert torch.all(out == torch.tensor([True, False])), "LukasiewiczLayer produce_weights_mask failed!"
        out = predicates._produce_weights_mask(torch.tensor([1.0, 0.5]), threshold=0.7)
        assert torch.all(out == torch.tensor([True, False])), "LukasiewiczLayer produce_weights_mask failed!"

        out = predicates._produce_weights_mask(torch.tensor([1.0, 1.0]), quantile=0.9)
        assert torch.all(out == torch.tensor([True, True])), "LukasiewiczLayer produce_weights_mask failed!"
        out = predicates._produce_weights_mask(torch.tensor([1.0, 1.0]), threshold=0.7)
        assert torch.all(out == torch.tensor([True, True])), "LukasiewiczLayer produce_weights_mask failed!"

        out = predicates._produce_weights_mask(torch.tensor([0.0, 0.0]), quantile=0.9)
        assert torch.all(out == torch.tensor([False, False])), "LukasiewiczLayer produce_weights_mask failed!"
        out = predicates._produce_weights_mask(torch.tensor([0.0, 0.0]), threshold=0.7)
        assert torch.all(out == torch.tensor([False, False])), "LukasiewiczLayer produce_weights_mask failed!"

        out = predicates._produce_weights_mask(torch.tensor([0.1, 0.1]), quantile=0.9)
        assert torch.all(out == torch.tensor([True, True])), "LukasiewiczLayer produce_weights_mask failed!"
        out = predicates._produce_weights_mask(torch.tensor([0.1, 0.1]), threshold=0.7)
        assert torch.all(out == torch.tensor([False, False])), "LukasiewiczLayer produce_weights_mask failed!"

    @staticmethod
    def test__predicates_explain():
        predicates = Predicates(feature_names=[f"Predicate{i}" for i in range(4)])

        # begin tests of different configurations

        # test 1: 50%.  Should remove the OR because we select only half of the input for that layer
        out = predicates.explain(
            quantile=0.5,
            parent_weights=torch.tensor([0.1, 0.2, 1.0, 5.0]),
            parent_mask=torch.tensor([0, 1, 2, 3]),
            explain_type='both',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            parent_logic_type='And',
            required_output_thresholds=torch.tensor(0.7)
        )
        assert out == ['Predicate2 >= 0.925', 'Predicate3 >= 0.985'], "LukasiewiczLayer did not explain correctly!"

        # test 2: 100%.  Should produce full logic, excluding the OR since the two inputs to the OR are the same.
        out = predicates.explain(
            quantile=1.0,
            parent_weights=torch.tensor([0.1, 0.2, 1.0, 5.0]),
            parent_mask=torch.tensor([0, 1, 2, 3]),
            explain_type='both',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            parent_logic_type='And',
            required_output_thresholds=torch.tensor(0.7)
        )
        assert out == ['Predicate0 >= 0.25', 'Predicate1 >= 0.625', 'Predicate2 >= 0.925', 'Predicate3 >= 0.985'], \
            "LukasiewiczLayer did not explain correctly!"

        # test 3: 50%.  Should produce half the logic with a negation
        out = predicates.explain(
            quantile=0.5,
            parent_weights=torch.tensor([0.1, 0.2, 1.0, -5.0]),
            parent_mask=torch.tensor([0, 1, 2, 3]),
            explain_type='both',
            input_features=torch.tensor([1.0, 1.0, 1.0, 0.0]),
            parent_logic_type='And',
            required_output_thresholds=torch.tensor(0.7)
        )
        assert out == ['NOT(Predicate3 >= 0.015)', 'Predicate2 >= 0.925'], \
            "LukasiewiczLayer did not explain correctly!"

        # test 4: 100%.  Should produce the full logic with the or and one of the OR inputs negated
        out = predicates.explain(
            quantile=1.0,
            parent_weights=torch.tensor([0.1, 0.2, 1.0, -5.0]),
            parent_mask=torch.tensor([0, 1, 2, 3]),
            explain_type='both',
            input_features=torch.tensor([1.0, 1.0, 1.0, 0.0]),
            parent_logic_type='And',
            required_output_thresholds=torch.tensor(0.7)
        )
        assert out == ['NOT(Predicate3 >= 0.015)', 'Predicate0 >= 0.25', 'Predicate1 >= 0.625', 'Predicate2 >= 0.925'],\
            "LukasiewiczLayer did not explain correctly!"

    @staticmethod
    def test__predicates_explain_sample():
        predicates = Predicates(feature_names=[f"Predicate{i}" for i in range(4)])

        # begin tests of different configurations

        # test 1: 50%.  Should remove the OR because we select only half of the input for that layer
        out = predicates.explain_sample(
            quantile=0.5,
            parent_weights=torch.tensor([0.1, 0.2, 1.0, 5.0]),
            parent_mask=torch.tensor([0, 1, 2, 3]),
            explain_type='both',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            parent_logic_type='And',
            required_output_thresholds=torch.tensor(0.7)
        )
        assert out == ['Predicate2 >= 0.7', 'Predicate3 >= 0.94'], "LukasiewiczLayer did not explain correctly!"

        # test 2: 100%.  Should produce full logic, excluding the OR since the two inputs to the OR are the same.
        out = predicates.explain_sample(
            quantile=1.0,
            parent_weights=torch.tensor([0.1, 0.2, 1.0, 5.0]),
            parent_mask=torch.tensor([0, 1, 2, 3]),
            explain_type='both',
            input_features=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            parent_logic_type='And',
            required_output_thresholds=torch.tensor(0.7)
        )
        assert out == ['Predicate0 >= 0.0', 'Predicate1 >= 0.0', 'Predicate2 >= 0.7', 'Predicate3 >= 0.94'], \
            "LukasiewiczLayer did not explain correctly!"

        # test 3: 50%.  Should produce half the logic with a negation
        out = predicates.explain_sample(
            quantile=0.5,
            parent_weights=torch.tensor([0.1, 0.2, 1.0, -5.0]),
            parent_mask=torch.tensor([0, 1, 2, 3]),
            explain_type='both',
            input_features=torch.tensor([1.0, 1.0, 1.0, 0.0]),
            parent_logic_type='And',
            required_output_thresholds=torch.tensor(0.7)
        )
        assert out == ['NOT(Predicate3 >= 0.06)', 'Predicate2 >= 0.7'], \
            "LukasiewiczLayer did not explain correctly!"

        # test 4: 100%.  Should produce the full logic with the or and one of the OR inputs negated
        out = predicates.explain_sample(
            quantile=1.0,
            parent_weights=torch.tensor([0.1, 0.2, 1.0, -5.0]),
            parent_mask=torch.tensor([0, 1, 2, 3]),
            explain_type='both',
            input_features=torch.tensor([1.0, 1.0, 1.0, 0.0]),
            parent_logic_type='And',
            required_output_thresholds=torch.tensor(0.7)
        )
        assert out == ['NOT(Predicate3 >= 0.06)', 'Predicate0 >= 0.0', 'Predicate1 >= 0.0', 'Predicate2 >= 0.7'], \
            "LukasiewiczLayer did not explain correctly!"

        # test 5: 50%.  Should produce only half the logic, but exclude the last feature which was below the required
        # threshold
        out = predicates.explain_sample(
            quantile=0.5,
            parent_weights=torch.tensor([0.1, 0.2, 1.0, -5.0]),
            parent_mask=torch.tensor([0, 1, 2, 3]),
            explain_type='both',
            input_features=torch.tensor([1.0, 1.0, 1.0, 0.5]),
            parent_logic_type='And',
            required_output_thresholds=torch.tensor(0.7)
        )
        assert out == ['Predicate2 >= 1.0'], \
            "LukasiewiczLayer did not explain correctly!"

        # test 6: 100%.  Should produce the full logic, but exclude the last feature which was below the required
        # threshold
        out = predicates.explain_sample(
            quantile=1.0,
            parent_weights=torch.tensor([0.1, 0.2, 1.0, -5.0]),
            parent_mask=torch.tensor([0, 1, 2, 3]),
            explain_type='both',
            input_features=torch.tensor([1.0, 1.0, 1.0, 0.5]),
            parent_logic_type='And',
            required_output_thresholds=torch.tensor(0.7)
        )
        assert out == ['Predicate0 >= 1.0', 'Predicate1 >= 1.0', 'Predicate2 >= 1.0'], \
            "LukasiewiczLayer did not explain correctly!"
