import os

from torchlogic.utils.explanations.simplification import Node, Explanation

from pytest import fixture

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(ROOT_DIR)


class TestNode:

    @fixture
    def node(self):
        pass
    
    @staticmethod
    def test__parse_explanation():
        pass # TestExplanation.test__parse_explanation() is testing the same

class TestExplanation:

    @staticmethod
    def test___text_to_phrases():
        text = """NOT ( OR(
 Longitude was greater than -122.36, 
NOT(AveOccup was less than or equal to 2.16338  )
) )
"""
        phrases = ['NOT', '(', 'OR', '(', 'Longitude was greater than -122.36', 'NOT', '(', 'AveOccup was less than or equal to 2.16338', ')', ')', ')']
        assert Explanation._text_to_phrases(text) == phrases

    @staticmethod
    def test___phrases_to_lines():
        phrases = ['OR', '(', 'AND', '(', 'NOT', '(', 'the compactness error is > 0.085', ')', 'the worst area is >= 221.819', 'the compactness error is >= 0.071', ')', 'AND', '(', 'the compactness error is >= 0.072', 'NOT', '(', 'the mean area is > 534.845', ')', ')', ')']
        lines = [('OR', 0), ('AND', 1), ('NOT', 2), ('the compactness error is > 0.085', 3), ('the worst area is >= 221.819', 2), ('the compactness error is >= 0.071', 2), ('AND', 1), ('the compactness error is >= 0.072', 2), ('NOT', 2), ('the mean area is > 534.845', 3)]
        assert Explanation._phrases_to_lines(phrases) == lines

    @staticmethod
    def test___line_to_node():
        line = ('OR', 0)
        assert Explanation._line_to_node(line) == Node("", 0, 'OR', 'logical')

        line = ('NOT', 2)
        assert Explanation._line_to_node(line) == Node("", 2, 'NOT', 'logical')

        line = ('the compactness error is < 0.085', 3)
        assert Explanation._line_to_node(line) == Node('the compactness error is < 0.085', 
                                                       3, "LEAF", 'logical')

    @staticmethod
    def test___text_to_tree():
        text = """OR(AND(NOT(the compactness error is > 0.085), the worst area is >= 221.819, the compactness error is >= 0.071), OR(the compactness error is >= 0.072,  NOT(the mean area is > 534.845)))"""

        root = Node("", 0, "OR", 'logical')

        level1_node1 = Node("", 1, "AND", 'logical')
        level1_node2 = Node("", 1, "OR", 'logical')

        level2_node11 = Node("", 2, "NOT", 'logical')
        level2_node12 = Node('the worst area is >= 221.819', 2, "LEAF", 'logical')
        level2_node13 = Node('the compactness error is >= 0.071', 2, "LEAF", 'logical')

        level3_node111 = Node('the compactness error is > 0.085', 3, "LEAF", 'logical')
        
        level2_node11.operands = [level3_node111]
        level1_node1.operands = [level2_node11, level2_node12, level2_node13]

        level2_node21 = Node('the compactness error is >= 0.072', 2, "LEAF", 'logical')
        level2_node22 = Node("", 2, "NOT", 'logical')

        level3_node221 = Node('the mean area is > 534.845', 3, "LEAF", 'logical')

        level2_node22.operands = [level3_node221]
        level1_node2.operands = [level2_node21, level2_node22]
        root.operands = [level1_node1, level1_node2]

        assert Explanation._text_to_tree(text, 'logical') == root

    @staticmethod
    def test___negate():
        node = Node("the number of pet's legs was 4", 5, "LEAF", 'logical')
        negated_node = Node("the number of pet's legs was 4", 5, "NOT", 'logical')
        assert Explanation._negate(node) == negated_node, "Didn't correctly negate LEAF with equality"

        node = Node("the number of pet's legs was greater than 3", 5, "LEAF", 'logical')
        negated_node = Node("the number of pet's legs was less than or equal to 3", 
                            5, "LEAF", 'logical')
        assert Explanation._negate(node) == negated_node, "Didn't correctly negate LEAF with inequality"

        node = Node("the number of pet's legs was 4", 5, "NOT", 'logical')
        negated_node = Node("the number of pet's legs was 4", 5, "LEAF", 'logical')
        assert Explanation._negate(node) == negated_node, "Didn't correctly negate NOT with equality"

        node = Node("the number of pet's legs was greater than 3", 5, "NOT", 'logical')
        negated_node = Node("the number of pet's legs was greater than 3", 5, "LEAF", 'logical')
        assert Explanation._negate(node) == negated_node, "Didn't correctly negate NOT with inequality"

        node = Node("", 3, "NOT", 'logical')
        node.operands = [Node("the number of goose's friends was 2", 4, "LEAF", 'logical')]
        negated_node = Node("the number of goose's friends was 2", 3, "LEAF", 'logical')
        assert Explanation._negate(node) == negated_node, "Didn't correctly negate NOT with operand and equality"

        node = Node("", 3, "NOT", 'logical')
        node.operands = [Node("the number of goose's friends was greater than 2", 
                              4, "LEAF", 'logical')]
        negated_node = Node("the number of goose's friends was greater than 2", 
                            3, "LEAF", 'logical')
        assert Explanation._negate(node) == negated_node, "Didn't correctly negate NOT with operand and equality"

        node = Node("", 3, "AND", 'logical')
        node.operands = [Node("the number of goose's friends was less than or equal to 2", 
                              4, "LEAF", 'logical'),
                         Node("the number of lion's gueses was greater than 17", 
                              4, "NOT", 'logical')]
        negated_node = Node("", 3, "OR", 'logical')
        negated_node.operands = [Node("the number of goose's friends was greater than 2", 
                                      4, "LEAF", 'logical'),
                                 Node("the number of lion's gueses was greater than 17", 
                                      4, "LEAF", 'logical')]
        assert Explanation._negate(node) == negated_node, "Didn't correctly negate AND"

        node = Node("", 3, "OR", 'logical')
        node.operands = [Node("the number of goose's friends was less than or equal to 2", 
                              4, "LEAF", 'logical'),
                         Node("the number of lion's gueses was greater than 17", 
                              4, "NOT", 'logical')]
        negated_node = Node("", 3, "AND", 'logical')
        negated_node.operands = [Node("the number of goose's friends was greater than 2", 
                                      4, "LEAF", 'logical'),
                                 Node("the number of lion's gueses was greater than 17", 
                                      4, "LEAF", 'logical')]
        assert Explanation._negate(node) == negated_node, "Didn't correctly negate OR"

    @staticmethod
    def test__push_negations_down():
        node = Node("the number of goose's friends was greater than 2", 4, "LEAF", 'logical')
        processed_node = Node("the number of goose's friends was greater than 2", 
                              4, "LEAF", 'logical')
        assert Explanation.push_negations_down(node) == processed_node, "Didn't correctly push negations down for LEAF"

        node = Node("the number of goose's friends was less than or equal to 2", 4, "NOT", 'logical')
        processed_node = Node("the number of goose's friends was greater than 2", 
                              4, "LEAF", 'logical')
        assert Explanation.push_negations_down(node) == processed_node, "Didn't correctly push negations down for LEAF"

        node = Node("", 3, "NOT", 'logical')
        node.operands = [Node("the number of goose's friends was less than 2", 4, "LEAF", 'logical')]
        processed_node = Node("the number of goose's friends was greater than or equal to 2", 
                              3, "LEAF", 'logical')
        assert Explanation.push_negations_down(node) == processed_node, "Didn't correctly push negations down for NOT with operand"

        node = Node("", 0, "NOT", 'logical')
        level1_node = Node("", 1, "NOT", 'logical')
        level2_node = Node("", 2, "NOT", 'logical')
        level2_node.operands = [Node("the number of goose's friends was less than 2", 
                                     3, "LEAF", 'logical')]
        level1_node.operands = [level2_node]
        node.operands = [level1_node]
        processed_node = Node("the number of goose's friends was greater than or equal to 2", 
                              0, "LEAF", 'logical')
        assert Explanation.push_negations_down(node) == processed_node, "Didn't correctly push negations down for NOT with operands"

        node = Node("", 0, "NOT", 'logical')
        level1_node = Node("", 1, "NOT", 'logical')
        level2_node = Node("", 2, "NOT", 'logical')
        level2_node.operands = [Node("the number of goose's friends was greater than 2", 
                                     3, "NOT", 'logical')]
        level1_node.operands = [level2_node]
        node.operands = [level1_node]
        processed_node = Node("the number of goose's friends was greater than 2", 
                              0, "LEAF", 'logical')
        assert Explanation.push_negations_down(node) == processed_node, "Didn't correctly push negations down for NOT with operands"

        node = Node("", 3, "AND", 'logical')
        node.operands = [Node("the number of goose's friends was greater than 2", 
                              4, "LEAF", 'logical'),
                         Node("the number of lion's guests was less than or equal to 17", 
                              4, "NOT", 'logical')]
        processed_node = Node("", 3, "AND", 'logical')
        processed_node.operands = [Node("the number of goose's friends was greater than 2", 
                                        4, "LEAF", 'logical'),
                                 Node("the number of lion's guests was greater than 17", 
                                      4, "LEAF", 'logical')]
        assert Explanation.push_negations_down(node) == processed_node, "Didn't correctly push negations down for AND"

    @staticmethod
    def test___collapse_consecutive_repeated_operands():
        node = Node("", 0, "AND", 'logical')
        level1_node = Node("", 1, "AND", 'logical')
        level2_node = Node("", 2, "AND", 'logical')
        level2_node.operands = [Node("the number of goose's friends was greater than 2", 
                                     3, "LEAF", 'logical'),
                                Node("the number of lion's guests was greater than 17", 
                                     3, "LEAF", 'logical')]
        level1_node.operands = [level2_node,
                                Node("the name of owl's friend is AE32", 2, "NOT", 'logical')]
        node.operands = [level1_node,
                         Node("tulip is happy", 1, "LEAF", 'logical')]
        processed_node = Node("", 0, "AND", 'logical')
        processed_node.operands = [Node("the number of goose's friends was greater than 2", 
                                        1, "LEAF", 'logical'),
                                   Node("the number of lion's guests was greater than 17", 
                                        1, "LEAF", 'logical'),
                                   Node("the name of owl's friend is AE32", 1, "NOT", 'logical'),
                                   Node("tulip is happy", 1, "LEAF", 'logical')]
        assert Explanation._collapse_consecutive_repeated_operands(node) == processed_node, "Didn't correctly collapse consecitive repeated operands for AND"

    @staticmethod
    def test__parse_explanation():
        node = Node(explanation='Longitude was greater than -122.36', level=1, 
                    logic_type='NOT', print_type="logical")
        node.parse_explanation()
        assert node.feature == "Longitude"
        assert node.sign == "greater than"
        assert node.value == "-122.36"

        node = Node(explanation='AveOccup was less than or equal to 2.64845', level=3, 
                    logic_type='NOT', print_type="logical")
        node.parse_explanation()
        assert node.feature == "AveOccup"
        assert node.sign == "less than or equal to"
        assert node.value == "2.64845"

        node = Node(explanation='the product on the opportunity being priced was from UT30_30CE3', 
                    level=0, logic_type='LEAF', print_type="logical")
        node.parse_explanation()
        assert node.feature == "the product on the opportunity being priced was from UT30_30CE3"
        assert node.sign == ''
        assert node.value == ''
