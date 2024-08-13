import re
from torchlogic.utils.explanations.simplification import Explanation


def remove_duplicate_words(input1: str):
    # Regex to matching repeated words
    regex = r'\b(\w+)(?:\W+\1\b)+'

    return re.sub(regex, r'\1', input1, flags=re.IGNORECASE)


def remove_character_combo(input1: str, char1: str, char2: str, replacement_char: str):
    # retplace the string ";," with ";"

    return input1.replace(f"{char1}{char2}", f"{replacement_char}")


def get_outputs(name, outputs):
    def hook(model, input, output):
        if isinstance(output, tuple):
            outputs[name] = output[0].detach()
        else:
            outputs[name] = output.detach()

    return hook


def register_hooks(model, outputs, mode='explanation'):
    for x in model.named_children():
        x[1].register_forward_hook(get_outputs(x[0], outputs))
        register_hooks(x[1], outputs)


def simplification(
        explanation_str,
        print_type,
        simplify,
        sample_level=False,
        verbose=False,
        ndigits: int = 3,
        exclusions: list[str] = None,
        min_max_feature_dict: dict = None,
        feature_importances: bool = False
):
    if verbose:
        print("\n_____\nInput string:\n\n", explanation_str)

    if simplify:
        assert print_type in ['logical', 'logical-natural'], \
            "print_type must be 'logical' or 'logical-natural' if simplify is True"
        
    if feature_importances:
        exp = Explanation(explanation_str, print_type)
        exp.root.sort_operands()
        if verbose:
            print("\n_____\nExplanation tree for feature importances (not simplified):\n", exp.root.tree_to_string())
        return exp.root

    if print_type == 'natural':
        return explanation_str
    
    exp = Explanation(explanation_str, print_type)
    if verbose:
        exp.root.sort_operands()
        print("\n_____\nExplanation tree:\n", exp.root.tree_to_string())

    if not simplify:
        exp.root.sort_operands()
        # return exp.root.tree_to_string()
        return exp.root
    
    exp.root = Explanation.push_negations_down(exp.root)
    if verbose:
        exp.root.sort_operands()
        print("\n_____\nNegations pushed down:\n\n", exp.root.tree_to_string())

    exp.root = Explanation.recursively_collapse_consecutive_repeated_operands(exp.root)
    if verbose:
        exp.root.sort_operands()
        print("\n_____\nConsecutive operands collapsed:\n", exp.root.tree_to_string())

    Explanation.recursive_explanation_parsing(exp.root)
    exp.root = Explanation.remove_redundant(exp.root)
    if verbose:
        exp.root.sort_operands()
        print("\n_____\nRedundant features removed:\n", exp.root.tree_to_string())

    exp.root = Explanation.recursively_collapse_single_operands(exp.root)
    if verbose:
        exp.root.sort_operands()
        print("\n_____\nSingle operands collapsed:\n", exp.root.tree_to_string())

    Explanation.recursive_explanation_parsing(exp.root)
    exp.root = Explanation.remove_redundant(exp.root)
    if verbose:
        exp.root.sort_operands()
        print("\n_____\nRedundant features removed:\n", exp.root.tree_to_string())

    if not sample_level:
        Explanation.recursive_explanation_parsing(exp.root) # Adding feature, sign, value to each node
        exp.simple_root = Explanation.create_between_explanations(exp.root)
        exp.simple_root.sort_operands()
        if verbose:
            print("\n_____\nBetween explanations created:\n", exp.simple_root.tree_to_string())
        # return exp.simple_root.tree_to_string() 
        return exp.simple_root 
    
    # Now we know that simplify is True, print_type is 'logical' and sample_level is True
    # Sample level simplification -- for sample level explanations only!
    exp.sample_root = Explanation.collapse_sample_explanation(exp.root)
    if verbose:
        exp.sample_root.sort_operands()
        print("\n_____\nThe tree collapsed to sample level:\n", exp.sample_root.tree_to_string())
    
    Explanation.recursive_explanation_parsing(exp.sample_root)
    exp.sample_root = Explanation.remove_redundant(exp.sample_root, verbose=verbose)
    if verbose:
        exp.sample_root.sort_operands()
        print("\n_____\nSample level redundant features removed:\n", exp.sample_root.tree_to_string())

    Explanation.recursive_explanation_parsing(exp.sample_root) # To add feature, sign, value to each node
    exp.sample_root = Explanation.create_between_explanations__sample_level(exp.sample_root)
    exp.sample_root.sort_operands()
    if verbose:
        print("\n_____\nSample level between explanations created:\n", exp.sample_root.tree_to_string())

    Explanation.format_explanation(exp, exp.sample_root, ndigits, exclusions, min_max_feature_dict)
    if verbose:
        print("\n_____\nSample level formatted explanations created:\n", exp.sample_root.tree_to_string())

    exp.sample_root.sort_operands()
    # return exp.sample_root.tree_to_string()
    return exp.sample_root

