import os

import pytest
from pytest import fixture

from torchlogic.utils.explanations import simplification

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(ROOT_DIR)


class TestBanditRRNTrainer:

    @fixture
    def two_layer_with_negations(self):
        return ("AND(OR(feature1 was >= 0.5, feature2 was >= 0.1234), "
                "OR(NOT(feature3 was category1), feature4 was >= 0.9))")

    @staticmethod
    def test__round_numeric_leaves(two_layer_with_negations):
        out = simplification(
            two_layer_with_negations,
            print_type='logical',
            simplify=True,
            sample_level=True,
            verbose=False,
            ndigits=2
        )
        assert out == ('\nAND '
                       '\n\tFeature1 was greater than or equal to 0.5'
                       '\n\tFeature2 was greater than or equal to 0.12'
                       '\n\tFeature4 was greater than or equal to 0.9'
                       '\n\tNOT feature3 was category1')

    @staticmethod
    def test__filter_excluded_leaves(two_layer_with_negations):
        out = simplification(
            two_layer_with_negations,
            print_type='logical',
            simplify=True,
            sample_level=True,
            verbose=False,
            ndigits=5,
            exclusions=['feature1 was']
        )
        assert out == ('\nAND '
                       '\n\tFeature2 was greater than or equal to 0.1234'
                       '\n\tFeature4 was greater than or equal to 0.9'
                       '\n\tNOT feature3 was category1')

        with pytest.raises(AssertionError) as e:
            _ = simplification(
                two_layer_with_negations,
                print_type='logical',
                simplify=True,
                sample_level=True,
                verbose=False,
                ndigits=5,
                exclusions=['feature1 was', 'feature2 was', 'feature3 was category1', 'feature4 was']
            )

    @staticmethod
    def test__simplification_logical_natural(two_layer_with_negations):
        out = simplification(
            two_layer_with_negations,
            print_type='logical-natural',
            simplify=True,
            sample_level=True,
            verbose=False,
            ndigits=5
        )
        assert out == ('\nAll the following are true: '
                       '\n\tFeature1 was greater than or equal to 0.5'
                       '\n\tFeature2 was greater than or equal to 0.1234'
                       '\n\tFeature4 was greater than or equal to 0.9'
                       '\n\tIt was not true that feature3 was category1')

        out = simplification(
            two_layer_with_negations,
            print_type='logical-natural',
            simplify=True,
            sample_level=False,
            verbose=False,
            ndigits=5
        )
        assert out == ('\nAll the following are true: '
                       '\n\tAny of the following are true: '
                       '\n\t\tFeature1 was greater than or equal to 0.5'
                       '\n\t\tFeature2 was greater than or equal to 0.1234'
                       '\n\tAny of the following are true: '
                       '\n\t\tFeature4 was greater than or equal to 0.9'
                       '\n\t\tIt was not true that feature3 was category1')

def test__simplification():
    
    explanation_str = """NOT(Longitude was greater than -122.36)"""
    output_str = "\n" + """Longitude was less than or equal to -122.36"""

    simplified_explanation_str = simplification(explanation_str, \
                                                print_type='logical', \
                                                simplify=True, \
                                                sample_level=False, \
                                                verbose=False)
    assert simplified_explanation_str.replace('\t', 4*' ') == output_str, "Didn't correctly produce negation"

    explanation_str = """NOT(OR(
Longitude was less than -122.36, 
NOT(AveOccup was less than or equal to 2.16338)
))"""
    output_str = """\nAND 
    AveOccup was less than or equal to 2.16338
    Longitude was greater than or equal to -122.36"""
    simplified_explanation_str = simplification(explanation_str, \
                                                print_type='logical', \
                                                simplify=True, \
                                                sample_level=False, \
                                                verbose=False)
    assert simplified_explanation_str.replace('\t', 4*' ') == output_str, "Didn't correctly simplify NOT OR"

    explanation_str = """NOT(AND(
Longitude was less than -122.36, 
NOT(AveOccup was less than or equal to 2.16338)
))"""
    output_str = """\nOR 
    AveOccup was less than or equal to 2.16338
    Longitude was greater than or equal to -122.36"""
    simplified_explanation_str = simplification(explanation_str, \
                                                print_type='logical', \
                                                simplify=True, \
                                                sample_level=False, \
                                                verbose=False)
    assert simplified_explanation_str.replace('\t', 4*' ') == output_str, "Didn't correctly simplify NOT AND"

    explanation_str = """NOT(OR(
AND(
NOT(OR(
Longitude was greater than -122.36, 
NOT(AveOccup was less than or equal to 2.16338), 
NOT(HouseAge was greater than 46.5), 
NOT(Latitude was less than or equal to 37.975), 
NOT(Longitude was less than or equal to -122.535), 
NOT(MedInc was greater than 2.78605), 
NOT(MedInc was greater than 5.14335)
))
)))
"""
    output_str = """\nOR 
    AveOccup was greater than 2.16338
    HouseAge was less than or equal to 46.5
    Latitude was greater than 37.975
    Longitude was greater than -122.535
    MedInc was less than or equal to 5.14335"""
    simplified_explanation_str = simplification(explanation_str, \
                                                print_type='logical', \
                                                simplify=True, \
                                                sample_level=False, \
                                                verbose=False)
    assert simplified_explanation_str.replace('\t', 4*' ') == output_str, "Didn't correctly simplify complicated case on global level"

    explanation_str = """NOT(OR(
AveOccup was greater than 2.5,
OR(NOT(AveOccup was greater than 2.15), MedInc was greater than or equal to 3.1),
AND(
HouseAge was greater than 50,
NOT(OR(
Longitude was greater than -122.36, 
NOT(AveOccup was less than or equal to 2.16338), 
NOT(HouseAge was greater than 46.5), 
OR(
NOT(Latitude was less than or equal to 37.975), 
MedInc was less than 3
)
)),
NOT(OR( 
NOT(Longitude was less than or equal to -122.535), 
AND(
NOT(MedInc was greater than 2.78605), 
NOT(MedInc was greater than 5.14335))
))
)
))
"""
    simplified_explanation_str = simplification(explanation_str, \
                                                print_type='natural', \
                                                simplify=False, \
                                                sample_level=False, \
                                                verbose=False)
    assert simplified_explanation_str == explanation_str, "Didn't correctly simplify for print_type='natural'"
    
    output_str = """\nAND 
    AveOccup was between 2.15 and 2.5
    MedInc was less than 3.1
    OR 
        AveOccup was greater than 2.16338
        HouseAge was less than or equal to 50.0
        Latitude was greater than 37.975
        Longitude was greater than -122.535
        MedInc was less than 3.0"""
    simplified_explanation_str = simplification(explanation_str, \
                                                print_type='logical', \
                                                simplify=True, \
                                                sample_level=False, \
                                                verbose=False)
    assert simplified_explanation_str.replace('\t', 4*' ') == output_str, "Didn't correctly simplify complicated case on global level"
    
    output_str = """\nAND 
    AveOccup was between 2.16 and 2.5
    HouseAge was less than or equal to 50
    Latitude was greater than 37.98
    Longitude was greater than -122.53
    MedInc was less than 3"""
    simplified_explanation_str = simplification(explanation_str, \
                                                print_type='logical', \
                                                simplify=True, \
                                                sample_level=True, \
                                                verbose=False, 
                                                ndigits=2)
    assert simplified_explanation_str.replace('\t', 4*' ') == output_str, "Didn't correctly simplify complicated case on sample level"
    
    