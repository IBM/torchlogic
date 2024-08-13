from .base import BasePredicates
import logging


class Predicates(BasePredicates):

    def __init__(self, feature_names: list):
        """
        Initialize a Predicates object.  The Predicates object is passed to the operands parameter of a
        LukasiewiczLayer or LukasiewiczChannelBlock and enables the explanation functionality of the torchlogic
        blocks, layers and modules.

        Example:
            feature_names = ['feature1', 'feature2', 'feature3', 'feature4']

            input_layer_and = LukasiewiczChannelAndBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes,
                n_selected_features=n_selected_features_input,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names)
                )

        Args:
            feature_names (list): A list of feature names.
        """
        super(Predicates, self).__init__(feature_names=feature_names)
        self.logger = logging.getLogger(self.__class__.__name__)
