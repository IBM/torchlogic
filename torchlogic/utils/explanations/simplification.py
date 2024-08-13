import numpy as np

# constants
GREATER_SIGNS = [">", ">=", "greater", "greater than", "greater than or equal", "greater than or equal to"]
GREATER_OR_EQUAL_SIGNS = [">=", "greater than or equal", "greater than or equal to"]
LESS_SIGNS = ["<", "<=", "less", "less than", "less than or equal", "less than or equal to"]
LESS_OR_EQUAL_SIGNS = ["<=", "less than or equal", "less than or equal to"]


class Node:
    def __init__(self, explanation, level, logic_type, print_type: str, importance_weight=np.nan, feature_importance=False):
        self.explanation = explanation.replace('"', "")
        self.level = level
        self.type = logic_type  # "AND", "OR", "NOT" or "LEAF"
        self.operands = []
        self.print_type = print_type # "logical", "logical-natural" or "natural" 
        assert logic_type in ["AND", "OR", "NOT", "LEAF"], "`logic_type` must be one of 'AND', 'OR', 'NOT', 'LEAF'."

        if self.type == "AND":
            self.natural_type = "all the following are true:"
        elif self.type == "OR":
            self.natural_type = "any of the following are true:"
        elif self.type == "NOT":
            self.natural_type = "it was NOT true that"
        elif self.type == "LEAF":
            self.natural_type = None

        self.feature_importance = feature_importance
        self.importance_weight = 1.
        if self.feature_importance:
            self.importance_weight = importance_weight

    def parse_explanation(self):
        self.feature, self.sign, self.value = Explanation.parse_explanation(self.explanation)
        if len(self.value) > 0:
            self.explanation = self.feature + " was " + self.sign + " " + str(self.value)
            self.explanation = self.explanation.replace("  ", " ")
        
    def add_operands(self, nodes):
        self._add_operand(nodes, 0)

    def _add_operand(self, nodes, i):
        for x in nodes[i:]:
            if x.level <= self.level:
                break

            if x.level == self.level + 1:  # is the node a operand
                self.operands.append(x)
                i = x._add_operand(nodes, i + 1)

        return i

    def print_tree(self):
        indentation = "\t" * self.level
        print(f"{indentation}{str(self)}")
        if self.operands:
            for operand in self.operands:
                operand.print_tree()
                
    def tree_to_string(self, input_string=''):
        indentation = "\t" * self.level
        if self.type == 'LEAF':
            self.parse_explanation()
        output_string = input_string + (f"\n{indentation}{str(self)}")
        if self.operands:
            for operand in self.operands:
                output_string = operand.tree_to_string(output_string)
        return output_string

    def decrease_level(self):
        self.level -= 1
        if self.operands:
            for operand in self.operands:
                operand.decrease_level()
        
    def simplify_and_and(self):
        total_operands = list()
        if not (self.type == "AND"):
            return 
        for child in self.operands:
            if not child.type == "AND":
                total_operands.append(child)
            else:
                for grandchild in child.operands:
                    grandchild.decrease_level()
                    total_operands.append(grandchild)
        self.operands = total_operands
          
    def simplify_or_or(self):
        total_operands = list() 
        if not (self.type == "OR"):
            return 
        for child in self.operands:
            if not child.type == "OR":
                total_operands.append(child)
            else:
                for grandchild in child.operands:
                    grandchild.decrease_level()
                    total_operands.append(grandchild)
        self.operands = total_operands
        
    def simplify_not_and(self):
        if not ((self.type == "NOT") and (len(self.operands)==1) and (self.operands[0].type == "AND")):
            return
        self.type = "OR"
        and_node = self.operands[0]
        self.explanation = and_node.explanation
        self.operands = [Explanation._negate_and_decrease_level(operand) for operand in and_node.operands]
        return

    def simplify_not_or(self):
        if not ((self.type == "NOT") and (len(self.operands)==1) and (self.operands[0].type == "OR")):
            return
        self.type = "AND"
        or_node = self.operands[0]
        self.explanation = or_node.explanation
        self.operands = [Explanation._negate_and_decrease_level(operand) for operand in or_node.operands]
        return
    
    def collapse_not(self):
        if not ((self.type == "NOT") and (len(self.operands)==1) and (self.operands[0].type == "LEAF")):
            return
        self.explanation = self.operands[0].explanation
        self.operands = []
        return
    
    def negate_explanation(self, substr1, substr2):
        if (len(self.operands) > 0) or (self.type != "NOT"):
            return
        if not substr1 in self.explanation:
            return
        self.explanation = self.explanation.replace(substr1, substr2)
        self.parse_explanation()    
        self.type = "LEAF"
        return
    
    def convert_not_to_leaf(self):
        self.collapse_not()
        self.negate_explanation("less than or equal to", "greater than")
        self.negate_explanation("less than", "greater than or equal to")
        self.negate_explanation("greater than or equal to", "less than")
        self.negate_explanation("greater than", "less than or equal to")
        self.negate_explanation(" <= ", " > ")
        self.negate_explanation(" >= ", " < ") 
        self.negate_explanation(" < ", " >= ")
        self.negate_explanation(" > ", " <= ")
        return
     
    def delete_duplicated_nodes(self):
        self.operands = list(set(self.operands))
        return
    
    def sort_operands(self):
        for child in self.operands:
            child.sort_operands()
        self.operands = Explanation.sort_nodes(self.operands)
        return 

    def get_feature_importances(self, parents_importance=1., importances_dict=None):
        if importances_dict is None:
            importances_dict = dict() 
        importance_weight = parents_importance * self.importance_weight
        if self.type == "LEAF":
            if self.explanation in importances_dict:
                importances_dict[self.explanation].append(importance_weight)
            else:
                importances_dict[self.explanation] = [importance_weight]
        else:
            for child in self.operands:
                child.get_feature_importances(importance_weight, importances_dict)
        return importances_dict
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        if (len(self.operands) < 1) and (len(other.operands) < 1) and \
            (self.level == other.level) and (self.explanation == other.explanation) and \
            (self.type == other.type):
            return True
        elif (len(self.operands) < 1) and (len(other.operands) < 1):
            return False
        return set(self.operands) == set(other.operands)

    def __str__(self) -> str:
        if self.feature_importance:
            if self.type == "LEAF":
                return f"{self.explanation[0].capitalize() + self.explanation[1:] + ' [' + str(self.importance_weight) + ']'}"
            if self.print_type == 'logical-natural':
                return f"{self.natural_type.capitalize()} {self.explanation}"
            return f"{self.type} {self.explanation} [{str(self.importance_weight)}]"
        else:
            if self.type == "LEAF":
                return f"{self.explanation[0].capitalize() + self.explanation[1:]}"
            if self.print_type == 'logical-natural':
                return f"{self.natural_type.capitalize()} {self.explanation}"
            return f"{self.type} {self.explanation}"

    def __repr__(self):
        return "Node(%s, %d, %s, %s)" % (self.explanation, self.level, self.type, self.print_type)
    
    def __hash__(self):
        return hash(self.__repr__())


class Explanation:
    def __init__(self, input_string, print_type):
        self.root = Explanation._text_to_tree(input_string, print_type)

    @staticmethod
    def _text_to_phrases(text):
        phrases = []
        phrase = ''
        for c in text:
            if not c in ['(', ')', ',']:
                phrase += c
            else:
                phrases.append(phrase)
                phrases.append(c)
                phrase = ''
        if not c in ['(', ')', ',']: # in case if there are no parentheses in the text
            phrases.append(phrase)
        phrases = [phrase.strip() for phrase in phrases]
        phrases = [phrase for phrase in phrases if len(phrase) > 0 and phrase != ',']
        return phrases
    
    @staticmethod
    def _phrases_to_lines(phrases):
        lines = []
        cur_level = 0

        for phrase in phrases:
            if not phrase in ['(', ')']:
                lines.append((phrase, cur_level))
            elif phrase == '(':
                cur_level += 1
            elif phrase == ')':
                cur_level -= 1
        return lines
    
    @staticmethod
    def _line_to_node(line, root=False, print_type:str = 'logical'):
        string, level = line
        if '[' in string:
            feature_importance = True
            words = [x.replace(']', '').strip() for x in string.split('[')]
            explanation_or_type = words[0]
            values = [np.abs(float(x)) for x in words[1:]]
            importance_weight = np.prod(values)
        else:
            feature_importance = False
            importance_weight = np.nan
            explanation_or_type = string
        
        if explanation_or_type in ["AND", "OR", "NOT"]:
            return Node("", level, explanation_or_type, print_type, importance_weight, feature_importance)
        else:
            return Node(explanation_or_type, level, "LEAF", print_type, importance_weight, feature_importance)

    @staticmethod
    def _text_to_tree(text, print_type):
        phrases = Explanation._text_to_phrases(text)
        lines = Explanation._phrases_to_lines(phrases)
        root = Explanation._line_to_node(lines[0], root=True, print_type=print_type)
        nodes = [Explanation._line_to_node(line, print_type=print_type) for line in lines[1:]]
        root.add_operands(nodes)
        return root
    
    @staticmethod
    def _negate(node):
        if node.type == "LEAF":
            new_node = Node(node.explanation, node.level, logic_type="NOT", print_type=node.print_type)
            new_node.convert_not_to_leaf()
            return new_node
        elif (node.type == "NOT") and (len(node.operands) < 1):
            node.parse_explanation()
            return Node(node.explanation, node.level, logic_type="LEAF", print_type=node.print_type)
        elif node.type == "NOT":
            child = node.operands[0]
            child.decrease_level()
            child = Explanation.push_negations_down(child)
            return child
        elif node.type == "AND":
            node.type = "OR"
            new_operands = [Explanation._negate(operand) for operand in node.operands]
            node.operands = new_operands
            return node
        elif node.type == "OR":
            node.type = "AND"
            new_operands = [Explanation._negate(operand) for operand in node.operands]
            node.operands = new_operands
            return node
        return Node('negation error', -1, None, None)
    
    @staticmethod
    def push_negations_down(node):
        if node.type == "LEAF":
            node.parse_explanation()
            return node
        elif (node.type == "NOT") and (len(node.operands) < 1):
            node.convert_not_to_leaf()
            return node
        elif node.type == "NOT":
            child = node.operands[0]
            new_node = Explanation._negate(child)
            new_node.decrease_level()
            return new_node
        else:
            new_node = Node(
                explanation=node.explanation, level=node.level, logic_type=node.type, print_type=node.print_type)
            new_operands = [Explanation.push_negations_down(child) for child in node.operands]
            new_node.operands = new_operands
            return new_node
               
    @staticmethod
    def _collapse_consecutive_repeated_operands(node):
        """Only works for AND and OR types of nodes, pushing negations down is needed before applying this"""
        node_type = node.type
        if not node_type in ["AND", "OR"]:
            return node
        same_type_children = list()
        different_type_children = list()
        for child in node.operands:
            if child.type == node_type:
                same_type_children.append(child)
            else:
                different_type_children.append(child)
        if len(same_type_children) < 1:
            return node
        new_children = different_type_children
        for child in same_type_children:
            child.decrease_level() # TODO replace to copy_and_decrease_level(child) to create a copy instead of changing the child
            new_children.extend(child.operands)
        new_node = Node(node.explanation, node.level, node.type, node.print_type)
        new_node.operands = new_children
        new_node = Explanation._collapse_consecutive_repeated_operands(new_node)
        return new_node

    @staticmethod
    def recursively_collapse_consecutive_repeated_operands(node):
        new_node = Explanation._collapse_consecutive_repeated_operands(node)
        new_operands = [Explanation.recursively_collapse_consecutive_repeated_operands(child) for child in new_node.operands]
        new_node.operands = new_operands
        return new_node
    
    @staticmethod
    def _negate_and_decrease_level(node):
        node.decrease_level()
        return Explanation._negate(node)
    
    @staticmethod
    def is_number(s):
        try:
            float(s) # for int, long and float
        except ValueError:
            try:
                complex(s) # for complex
            except ValueError:
                return False
        return True

    @staticmethod
    def is_integer(s):
        try:
            int(s)  # for int
        except ValueError:
            return False
        return True

    def is_integer_as_float(self, s):
        if self.is_number(s) and not self.is_integer(s):
            potential_integer = float(s)
            rounded_potential_integer = np.round(potential_integer, 0)
            if np.isclose(rounded_potential_integer, potential_integer):
                return True
        return False

    @staticmethod
    def _last_word_is_number(s):
        split = s.rsplit(' ', 1)
        if len(split) < 2:
            return (s, False)
        last_word = split[1]
        is_number = Explanation.is_number(last_word)
        return (last_word, is_number)

    @staticmethod
    def parse_explanation(x):
        """Parse a single explanation to a tuple (featuer, sign, value)"""
        last_word, is_number = Explanation._last_word_is_number(x)
        if not is_number:
            return (x, '', '')

        feature_sign = x.split(last_word)[0]

        for sep in [" is ", " was "]:
            if sep in feature_sign:
                feature_sign = feature_sign.split(sep)
        if len(feature_sign) != 2:
            print ("Complicated explanation parsing case causes error:", x)
            return (x, '', '')
        feature = feature_sign[0].strip()
        sign = feature_sign[1].strip()
        return feature, sign, last_word
    
    @staticmethod
    def recursive_explanation_parsing(node): 
        """Parse the node's explanation and all it's operands' explanations recursively"""
        node.parse_explanation()
        for child in node.operands:
            Explanation.recursive_explanation_parsing(child)

    @staticmethod
    def collapse_sample_explanation(node):
        if len(node.operands) < 1:
            return node
        
        total_operands = list()
        queue = [node]
        while queue:
            cur_node = queue.pop(0)
            for child in cur_node.operands:
                if len(child.operands) < 1:
                    new_leaf = Node(child.explanation, 1, child.type, node.print_type)
                    total_operands.append(new_leaf)
                else:
                    queue.append(child)
        final_node = Node('', 0, "AND", node.print_type)
        final_node.operands = list(set(total_operands))
        return final_node
    
    @staticmethod
    def _group_operands(node):
        feature_dict = dict()
        for operand in node.operands:
            feature = operand.feature
            if feature in feature_dict:
                feature_dict[feature].append(operand)
            else:
                feature_dict[feature] = [operand]
        return feature_dict

    @staticmethod
    def _remove_redundant_for_feature(parent_type, node_list, verbose=False):
        if not parent_type in ["AND", "OR"]:
            return node_list
        
        sign_dict = dict()
        output_list = list()
        
        for node in node_list:
            # Convert sign to its meaning (greater or less)
            sign = node.sign
            if sign in GREATER_OR_EQUAL_SIGNS:
                sign_meaning = "greater than"
                node.sign = "greater than or equal to"
            elif sign in GREATER_SIGNS:
                sign_meaning = "greater than"
                node.sign = "greater than"
            elif sign in LESS_OR_EQUAL_SIGNS:
                sign_meaning = "less than"
                node.sign = "less than or equal to"
            elif sign in LESS_SIGNS:
                sign_meaning = "less than"
                node.sign = "less than"
            else:
                output_list.append(node)
                if sign != '':
                    print("ATTENTION! Unknown sign detected: ", sign)
                continue
                
            # Adding node to sign_dict based on its sign meaning
            if sign_meaning in sign_dict:
                sign_dict[sign_meaning].append(node)
            else:
                sign_dict[sign_meaning] = [node]
                
        for sign in sign_dict:
            values = [(float(node.value), node.sign) for node in sign_dict[sign]]
            values.sort(key=lambda t: t[0], reverse=True) # descending order
            if sign in ["greater than or equal to","greater than"]:
                if parent_type == "AND":
                    final_tuple = values[0]
                elif parent_type == "OR":
                    final_tuple = values[-1]
                else:
                    print("Unknown parent type received: ", parent_type)
                    final_value = np.nan
            elif sign in ["less than or equal to", "less than"]:
                if parent_type == "AND":
                    final_tuple = values[-1]
                elif parent_type == "OR":
                    final_tuple = values[0]
                else:
                    print("Unknown parent type received: ", parent_type)
                    final_value = np.nan
            else:
                final_value = np.nan
                if sign != '':
                    print("ATTENTION! Unknown sign detected: ", sign, " sign len:", len(sign))
                output_list.extend(sign_dict[sign])
                continue
            # TODO check for contraditions like "between 10 and 1"
            final_value = final_tuple[0]
            final_sign = final_tuple[1]
            new_explanation = sign_dict[sign][0].feature + " was " + final_sign + " " + str(final_value)
            level = sign_dict[sign][0].level
            new_node = Node(new_explanation, level, "LEAF", node.print_type)
            output_list.append(new_node)
            
        if verbose:
            if len(output_list) < len(node_list):
                print("Redundancy removed from ")
                print(*node_list, sep='\n')
                print(" to ")
                print(*output_list, sep='\n')
            
        return output_list
    
    @staticmethod
    def remove_redundant(node, verbose=False):
        feature_dict = Explanation._group_operands(node)
        if len(feature_dict) < 1:
            return node
        
        total_operands = list()
        for feature in feature_dict:
            
            if feature == '':
                for deep_node in feature_dict['']:
                    total_operands.append(Explanation.remove_redundant(deep_node))
                continue
            
            simplified_operands = Explanation._remove_redundant_for_feature(node.type, \
                                                                            feature_dict[feature], verbose)
            total_operands.extend(simplified_operands)

        node.operands = total_operands
        return node
    
    @staticmethod
    def collapse_single_operand(node):
        if not node.type in ["AND", "OR"]:
            return node
        
        if len(node.operands) != 1:
            return node
        
        new_node = node.operands[0]
        new_node.decrease_level()
        return new_node
    
    @staticmethod
    def recursively_collapse_single_operands(node):
        final_operands = [Explanation.recursively_collapse_single_operands(operand) for \
                          operand in node.operands]
        node.operands = final_operands
        return Explanation.collapse_single_operand(node)
    
    @staticmethod
    def _create_between_explanation_for_AND(node_list):
        if len(node_list) != 2:
            return node_list
        # TODO check if both are leaves, not NOTs etc; check levels?
        feature = node_list[0].feature
        level = node_list[0].level
        sign1 = node_list[0].sign
        sign2 = node_list[1].sign
        value1 = node_list[0].value
        value2 = node_list[1].value
        
        if (sign1 in GREATER_SIGNS) and (sign2 in LESS_SIGNS):
            new_sign = "between"
            new_explanation = feature + " was between " + str(value1) + " and " + str(value2)
            new_node = Node(new_explanation, level, "LEAF", node_list[0].print_type)
            return [new_node]
        
        if (sign2 in GREATER_SIGNS) and (sign1 in LESS_SIGNS):
            new_sign = "between"
            new_explanation = feature + " was between " + str(value2) + " and " + str(value1)
            new_node = Node(new_explanation, level, "LEAF", node_list[0].print_type)
            return [new_node]
        
        return node_list
    
    @staticmethod
    def create_between_explanations__sample_level(node):
        """Only works for simplified tree of a sample explanation, 
        i.e. AND node with a single level of operands"""
        feature_dict = Explanation._group_operands(node)
        if len(feature_dict) < 1:
            return node
        
        total_operands = list()
        for feature in feature_dict:
            if len(feature_dict[feature]) == 1:
                total_operands.extend(feature_dict[feature])
            elif len(feature_dict[feature]) == 2:
                output = Explanation._create_between_explanation_for_AND(feature_dict[feature])
                total_operands.extend(output)
            else:
                total_operands.extend(feature_dict[feature])
        new_node = Node("", 0, "AND", node.print_type)
        new_node.operands = total_operands
        return new_node

    @staticmethod
    def create_between_explanations(node):
        feature_dict = Explanation._group_operands(node)
        if len(feature_dict) < 1:
            return node
        
        total_operands = list()
        if node.type != "AND":
            
            for feature in feature_dict:
                
                if feature == '':
                    for deep_node in feature_dict['']:
                        total_operands.append(Explanation.create_between_explanations(deep_node))
                    continue
                else:
                    total_operands.extend(feature_dict[feature])
            
            new_node = Node(node.explanation, node.level, node.type, node.print_type)
            new_node.operands = total_operands
            return new_node
            
        for feature in feature_dict:
            
            if feature == '':
                for deep_node in feature_dict['']:
                    total_operands.append(Explanation.create_between_explanations(deep_node))
                continue

            if len(feature_dict[feature]) == 1:
                total_operands.extend(feature_dict[feature])
            elif len(feature_dict[feature]) == 2:
                output = Explanation._create_between_explanation_for_AND(feature_dict[feature])
                total_operands.extend(output)
            else:
                total_operands.extend(feature_dict[feature])
        
        new_node = Node(node.explanation, node.level, node.type, node.print_type)
        new_node.operands = total_operands
        return new_node 

    @staticmethod
    def sort_node_types(node_type):
        if node_type == 'LEAF':
            return 1
        if node_type == 'NOT':
            return 2
        if node_type == 'AND':
            return 3
        return 4

    @staticmethod
    def sort_nodes(l):
        if len(l) < 1:
            return l
        expanded_list = [(node, Explanation.sort_node_types(node.type), node.explanation) for node in l]
        sorted_list = sorted(expanded_list, key=lambda x: (x[1], x[2]))
        return [t[0] for t in sorted_list]

    def format_explanation(self, node, ndigits, exclusions: list[str] = None, min_max_feature_dict: dict = None):
        new_operands = []
        if node.operands:
            for operand in node.operands:
                if operand.type == "LEAF" or ((operand.type == "NOT") and (len(operand.operands) < 1)):
                    self._round_numeric_leaves(operand, ndigits)
                    if min_max_feature_dict is not None:
                        self._add_percentile(operand, min_max_feature_dict)
                    if exclusions is not None:
                        new_operand = self._filter_excluded_leaf(operand, exclusions)
                    else:
                        new_operand = operand
                    if new_operand is not None:
                        new_operands += [new_operand]
                else:
                    new_operands += [operand]
            if new_operands:
                node.operands = new_operands
            else:
                raise AssertionError("exclusions removed all features from logic!")
        else:
            return node.operands

    def _round_numeric_leaves(self, node, ndigits):
        tokens = node.explanation.split(" ")
        tokens = [str(int(token) if self.is_integer(token)
                      else (int(np.round(float(token), 0)) if self.is_integer_as_float(token)
                            else round(float(token), ndigits=ndigits)))
                  if self.is_number(token) else token for token in tokens]
        node.explanation = " ".join(tokens)

    @staticmethod
    def _filter_excluded_leaf(node, exclusions: list[str]):
        for exclusion in exclusions:
            print(node.explanation, exclusion)
            if node.explanation.find(exclusion) > -1:
                return None
        return node

    @staticmethod
    def _add_percentile(node, min_max_feature_dict: dict):
        def _suffix(ptile):
            if ptile == '1':
                suffix = 'st'
            elif ptile == '2':
                suffix = 'nd'
            elif ptile == '3':
                suffix = 'rd'
            else:
                suffix = 'th'
            return suffix

        if node.type == 'LEAF':
            for k in min_max_feature_dict.keys():
                if node.explanation.find(k) > -1:
                    if node.explanation.find("between") > -1:
                        exp_parse = node.explanation.split(" between ")
                        exp_parse1 = [exp_parse[0]]
                        exp_parse2 = exp_parse[-1].split(" ")
                        num_1 = exp_parse2[0]
                        num_2 = exp_parse2[2]
                        ptile_1 = str(int((float(num_1) - min_max_feature_dict[k]['min'])
                                   / (min_max_feature_dict[k]['max'] - min_max_feature_dict[k]['min']) * 100))
                        ptile_2 = str(int((float(num_2) - min_max_feature_dict[k]['min'])
                                   / (min_max_feature_dict[k]['max'] - min_max_feature_dict[k]['min']) * 100))
                        node.explanation = " ".join(exp_parse1 + ["between"] + [num_1] + ["and"] + [num_2]
                                                    + [f"({ptile_1 + _suffix(ptile_1)} "
                                                       f"and {ptile_2 + _suffix(ptile_2)} percentiles)"])
                    else:
                        exp_parse = node.explanation.split(" ")
                        num = exp_parse[-1]
                        ptile = str(int((float(num) - min_max_feature_dict[k]['min'])
                                 / (min_max_feature_dict[k]['max'] - min_max_feature_dict[k]['min']) * 100))
                        node.explanation = " ".join(exp_parse[:-1] + [num] + [f"({ptile + _suffix(ptile)} percentile)"])
    

__all__ = [Node, Explanation]