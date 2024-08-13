from typing import Union
import logging

import torch
from ..utils.operations import val_clamp
from torchlogic.nn.base import (LukasiewiczChannelBlock, BasePredicates,
                                VariationalLukasiewiczChannelBlock, AttentionLukasiewiczChannelBlock)


class LukasiewiczChannelAndBlock(LukasiewiczChannelBlock):

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands: Union[LukasiewiczChannelBlock, BasePredicates],
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = 0.2
    ):
        """
        A logical AND channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
        """
        super(LukasiewiczChannelAndBlock, self).__init__(
            channels, in_features, out_features, n_selected_features,
            parent_weights_dimension, operands, 'And', outputs_key, add_negations, weight_init)
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor):
        """
        A channel logical and.

        Args:
            x: intput tensor [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]

        Returns:
            out: output tensor [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        """
        # X: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        # WEIGHTS: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        # MASK: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 3:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
        if x.ndim == 4:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)

        # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
        x = x.gather(-1, self.mask.unsqueeze(0).expand(x.size(0), *self.mask.shape).unsqueeze(-2))

        # explicit negation
        # WEIGHTS AFTER TRANSPOSE: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES, 1]
        weights = self.weights.transpose(-2, -1).unsqueeze(-1)
        weights_sign = weights.data.detach().clone().greater_equal(torch.tensor(0.0)).float()
        abs_weights = torch.abs(weights)
        pos = (1 - x) @ (abs_weights * weights_sign)
        neg = x @ (abs_weights * (1 - weights_sign))
        tot = pos + neg
        out = val_clamp(self.bias - tot)

        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        out = out.squeeze(-1).transpose(-2, -1)

        return out


class LukasiewiczChannelOrBlock(LukasiewiczChannelBlock):

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands: Union[LukasiewiczChannelBlock, BasePredicates],
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = 0.2
    ):
        """
        A logical OR channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
        """
        super(LukasiewiczChannelOrBlock, self).__init__(
            channels, in_features, out_features, n_selected_features,
            parent_weights_dimension, operands, 'Or', outputs_key, add_negations, weight_init)

    def forward(self, x: torch.Tensor):
        """
        A channel logical or.

        Args:
            x: intput tensor [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]

        Returns:
            out: output tensor [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        """
        # X: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        # WEIGHTS: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        # MASK: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 3:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
        if x.ndim == 4:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)

        # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
        x = x.gather(-1, self.mask.unsqueeze(0).expand(x.size(0), *self.mask.shape).unsqueeze(-2))

        # explicit negation
        # WEIGHTS AFTER TRANSPOSE: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES, 1]
        weights = self.weights.transpose(-2, -1).unsqueeze(-1)
        weights_sign = weights.data.detach().clone().greater_equal(torch.tensor(0.0)).float()
        abs_weights = torch.abs(weights)
        pos = x @ (abs_weights * weights_sign)
        neg = (1 - x) @ (abs_weights * (1 - weights_sign))
        tot = pos + neg
        out = val_clamp(1 - self.bias + tot)

        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        out = out.squeeze(-1).transpose(-2, -1)

        return out


class LukasiewiczChannelXOrBlock(LukasiewiczChannelBlock):

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands: Union[LukasiewiczChannelBlock, BasePredicates],
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = 0.2
    ):
        """
        A logical XOR channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
        """
        super(LukasiewiczChannelXOrBlock, self).__init__(
            channels, in_features, out_features, n_selected_features,
            parent_weights_dimension, operands, 'XOr', outputs_key, add_negations, weight_init)

    def forward(self, x: torch.Tensor):
        """
        A channel logical xor.

        Args:
            x: intput tensor [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]

        Returns:
            out: output tensor [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        """
        # X: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        # WEIGHTS: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        # MASK: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 3:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
        if x.ndim == 4:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)

        # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
        x = x.gather(-1, self.mask.unsqueeze(0).expand(x.size(0), *self.mask.shape).unsqueeze(-2))

        # explicit negation
        # WEIGHTS AFTER TRANSPOSE: [CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
        weights = self.weights.transpose(-2, -1).unsqueeze(-2)
        weights_sign = weights.data.detach().clone().greater_equal(torch.tensor(0.0)).float()
        abs_weights = torch.abs(weights)
        pos = (abs_weights * weights_sign) * (1.0 - x)
        neg = (abs_weights * (1.0 - weights_sign)) * x
        tot = pos + neg

        # ensure one true statement
        out_0 = (1.0 - tot)
        out_1 = val_clamp(out_0.gather(-1, out_0.argmax(dim=-1).unsqueeze(-1)))
        out_2 = val_clamp(out_0.sum(dim=-1).unsqueeze(-1) - 1.0)
        out = out_1 - out_2

        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        out = out.squeeze(-1).transpose(-2, -1)

        return val_clamp(out)


class VariationalLukasiewiczChannelAndBlock(VariationalLukasiewiczChannelBlock):

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands: Union[LukasiewiczChannelBlock, BasePredicates],
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = 0.2,
            var_emb_dim: int = 50,
            var_n_layers: int = 2
    ):
        """
        A logical AND channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
        """
        super(VariationalLukasiewiczChannelAndBlock, self).__init__(
            channels, in_features, out_features, n_selected_features,
            parent_weights_dimension, operands, 'And', outputs_key, add_negations, weight_init,
            var_emb_dim, var_n_layers)
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor):
        """
        A channel logical and.

        Args:
            x: intput tensor [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]

        Returns:
            out: output tensor [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        """
        # X: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        # WEIGHTS: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        # MASK: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 3:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
        if x.ndim == 4:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)

        # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
        x = x.gather(-1, self.mask.unsqueeze(0).expand(x.size(0), *self.mask.shape).unsqueeze(-2))

        # variational sparsity
        attn = self.sample_mask()

        # explicit negation
        # WEIGHTS AFTER TRANSPOSE: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES, 1]
        weights = (self.weights * attn).transpose(-2, -1).unsqueeze(-1)
        weights_sign = weights.data.detach().clone().greater_equal(torch.tensor(0.0)).float()
        abs_weights = torch.abs(weights)
        pos = (1 - x) @ (abs_weights * weights_sign)
        neg = x @ (abs_weights * (1 - weights_sign))
        tot = pos + neg
        out = val_clamp(self.bias - tot)

        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        out = out.squeeze(-1).transpose(-2, -1)

        return out


class VariationalLukasiewiczChannelOrBlock(VariationalLukasiewiczChannelBlock):

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands: Union[LukasiewiczChannelBlock, BasePredicates],
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = 0.2,
            var_emb_dim: int = 50,
            var_n_layers: int = 2
    ):
        """
        A logical OR channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
        """
        super(VariationalLukasiewiczChannelOrBlock, self).__init__(
            channels, in_features, out_features, n_selected_features,
            parent_weights_dimension, operands, 'Or', outputs_key, add_negations, weight_init,
            var_emb_dim, var_n_layers)

    def forward(self, x: torch.Tensor):
        """
        A channel logical or.

        Args:
            x: intput tensor [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]

        Returns:
            out: output tensor [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        """
        # X: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        # WEIGHTS: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        # MASK: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 3:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
        if x.ndim == 4:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)

        # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
        x = x.gather(-1, self.mask.unsqueeze(0).expand(x.size(0), *self.mask.shape).unsqueeze(-2))

        # variational sparsity
        attn = self.sample_mask()

        # explicit negation
        # WEIGHTS AFTER TRANSPOSE: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES, 1]
        weights = (self.weights * attn).transpose(-2, -1).unsqueeze(-1)
        weights_sign = weights.data.detach().clone().greater_equal(torch.tensor(0.0)).float()
        abs_weights = torch.abs(weights)
        pos = x @ (abs_weights * weights_sign)
        neg = (1 - x) @ (abs_weights * (1 - weights_sign))
        tot = pos + neg
        out = val_clamp(1 - self.bias + tot)

        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        out = out.squeeze(-1).transpose(-2, -1)

        return out


class VariationalLukasiewiczChannelXOrBlock(VariationalLukasiewiczChannelBlock):

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands: Union[LukasiewiczChannelBlock, BasePredicates],
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = 0.2,
            var_emb_dim: int = 50,
            var_n_layers: int = 2
    ):
        """
        A logical XOR channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
        """
        super(VariationalLukasiewiczChannelXOrBlock, self).__init__(
            channels, in_features, out_features, n_selected_features,
            parent_weights_dimension, operands, 'XOr', outputs_key, add_negations, weight_init,
            var_emb_dim, var_n_layers)

    def forward(self, x: torch.Tensor):
        """
        A channel logical xor.

        Args:
            x: intput tensor [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]

        Returns:
            out: output tensor [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        """
        # variational sparsity
        attn = self.sample_mask()

        # WEIGHTS AFTER TRANSPOSE: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES, 1]
        weights = (self.weights * attn).transpose(-2, -1).unsqueeze(-1)
        weights_sign = weights.data.detach().clone().greater_equal(torch.tensor(0.0)).float()
        abs_weights = torch.abs(weights)

        # Disjunction
        x_or = x.clone()
        
        # X: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        # WEIGHTS: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        # MASK: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        if x_or.ndim == 2:
            x_or = x_or.unsqueeze(1)
        if x_or.ndim == 3:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x_or = x_or.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
        if x_or.ndim == 4:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x_or = x_or.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)

        # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
        x_or = x_or.gather(-1, self.mask.unsqueeze(0).expand(x_or.size(0), *self.mask.shape).unsqueeze(-2))

        # explicit negation
        pos = x_or @ (abs_weights * weights_sign)
        neg = (1 - x_or) @ (abs_weights * (1 - weights_sign))
        tot = pos + neg
        out_or = val_clamp(1 - self.bias + tot)

        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        out_or = out_or.squeeze(-1).transpose(-2, -1)

        # Conjunction
        x_and = x.clone()
        # X: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        # WEIGHTS: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        # MASK: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        if x_and.ndim == 2:
            x_and = x_and.unsqueeze(1)
        if x_and.ndim == 3:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x_and = x_and.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
        if x_and.ndim == 4:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x_and = x_and.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)

        # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
        x_and = x_and.gather(-1, self.mask.unsqueeze(0).expand(x_and.size(0), *self.mask.shape).unsqueeze(-2))

        pos = (1 - x_and) @ (abs_weights * weights_sign)
        neg = x_and @ (abs_weights * (1 - weights_sign))
        tot = pos + neg
        out_and = val_clamp(self.bias - tot)

        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        out_and = out_and.squeeze(-1).transpose(-2, -1)

        return val_clamp(out_or - out_and)


# class SparseLukasiewiczChannelXOrBlock(SparseLukasiewiczChannelBlock):
#
#     def __init__(
#             self,
#             channels: int,
#             in_features: int,
#             out_features: int,
#             n_selected_features: int,
#             parent_weights_dimension: str,
#             operands: Union[LukasiewiczLayer, LukasiewiczChannelBlock, BasePredicates],
#             outputs_key: str,
#             add_negations: bool = False,
#             weight_init: float = 0.2,
#             tau: float = 0.2,
#             attn_emb_dim: int = 50
#     ):
#         """
#         A logical XOR channel block.
#
#         Args:
#             channels (int): The number of versions of logic.
#             in_features (int): The number of features in total from the previous layer or input.
#             out_features (int): The number of output features, corresponding to the number of AND nodes for output.
#             n_selected_features (int): The number of features to use for input to logic from in_features.
#             parent_weights_dimension (str): One of 'out_features', 'channels'.
#             operands (Union[LukasiewiczLayer, LukasiewiczChannelBlock, BasePredicates]): The child logic.
#             outputs_key (str): name of outputs used during explanation generation.
#             add_negations (bool): add negated logic matching original logics.
#         """
#         super(SparseLukasiewiczChannelXOrBlock, self).__init__(
#             channels, in_features, out_features, n_selected_features,
#             parent_weights_dimension, operands, 'XOr', outputs_key, add_negations, weight_init,
#             tau, attn_emb_dim)
#
#     def forward(self, x: torch.Tensor):
#         """
#         A channel logical xor.
#
#         Args:
#             x: intput tensor [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
#
#         Returns:
#             out: output tensor [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
#         """
#         # X: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
#         # WEIGHTS: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
#         # MASK: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
#         if x.ndim == 2:
#             x = x.unsqueeze(1)
#         if x.ndim == 3:
#             # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
#             x = x.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
#         if x.ndim == 4:
#             # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
#             x = x.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)
#
#         # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
#         x = x.gather(-1, self.mask.unsqueeze(0).expand(x.size(0), *self.mask.shape).unsqueeze(-2))
#
#         # variational sparsity
#         attn = self.sample_attn()
#
#         # explicit negation
#         # WEIGHTS AFTER TRANSPOSE: [CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
#         weights = (self.weights * attn).transpose(-2, -1).unsqueeze(-2)
#         weights_sign = weights.data.detach().clone().greater_equal(torch.tensor(0.0)).float()
#         abs_weights = torch.abs(weights)
#         pos = (abs_weights * weights_sign) * (1.0 - x)
#         neg = (abs_weights * (1.0 - weights_sign)) * x
#         tot = pos + neg
#
#         # ensure one true statement
#         out_0 = (1.0 - tot)
#         out_1 = val_clamp(out_0.gather(-1, out_0.argmax(dim=-1).unsqueeze(-1)))
#         out_2 = val_clamp(out_0.sum(dim=-1).unsqueeze(-1) - 1.0)
#         out = out_1 - out_2
#
#         # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
#         out = out.squeeze(-1).transpose(-2, -1)
#
#         return val_clamp(out)


class AttentionLukasiewiczChannelAndBlock(AttentionLukasiewiczChannelBlock):

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands: Union[LukasiewiczChannelBlock, BasePredicates],
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = 0.2,
            attn_emb_dim: int = 32,
            attn_n_layers: int = 2
    ):
        """
        A logical AND channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
        """
        super(AttentionLukasiewiczChannelAndBlock, self).__init__(
            channels, in_features, out_features, n_selected_features,
            parent_weights_dimension, operands, 'And', outputs_key, add_negations, weight_init,
            attn_emb_dim, attn_n_layers)
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor):
        """
        A channel logical and.

        Args:
            x: intput tensor [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]

        Returns:
            out: output tensor [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        """
        # X: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        # WEIGHTS: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        # MASK: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 3:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
        if x.ndim == 4:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)

        # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
        x = x.gather(-1, self.mask.unsqueeze(0).expand(x.size(0), *self.mask.shape).unsqueeze(-2))

        # variational sparsity
        weights_sign_attn = self.weights.data.detach().clone().repeat(
            x.size(0), 1, 1, 1).transpose(-2, -1).unsqueeze(-2).greater_equal(torch.tensor(0.0)).float()
        # pos_attn = (1 - x) * weights_sign_attn
        # neg_attn = x * (1 - weights_sign_attn)
        # x_attn = pos_attn + neg_attn
        attn = self.sample_mask(torch.cat([x, weights_sign_attn], dim=-1))
        attn = attn.squeeze(-2).transpose(-2, -1)
        weights = self.weights.repeat(x.size(0), 1, 1, 1)

        # explicit negation
        # WEIGHTS AFTER TRANSPOSE: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES, 1]
        weights = (weights * attn).transpose(-2, -1).unsqueeze(-1)
        weights_sign = weights.data.detach().clone().greater_equal(torch.tensor(0.0)).float()
        abs_weights = torch.abs(weights)
        pos = (1 - x) @ (abs_weights * weights_sign)
        neg = x @ (abs_weights * (1 - weights_sign))
        tot = pos + neg
        out = val_clamp(self.bias - tot)

        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        out = out.squeeze(-1).transpose(-2, -1)

        return out


class AttentionLukasiewiczChannelOrBlock(AttentionLukasiewiczChannelBlock):

    def __init__(
            self,
            channels: int,
            in_features: int,
            out_features: int,
            n_selected_features: int,
            parent_weights_dimension: str,
            operands: Union[LukasiewiczChannelBlock, BasePredicates],
            outputs_key: str,
            add_negations: bool = False,
            weight_init: float = 0.2,
            attn_emb_dim: int = 32,
            attn_n_layers: int = 2
    ):
        """
        A logical OR channel block.

        Args:
            channels (int): The number of versions of logic.
            in_features (int): The number of features in total from the previous layer or input.
            out_features (int): The number of output features, corresponding to the number of AND nodes for output.
            n_selected_features (int): The number of features to use for input to logic from in_features.
            parent_weights_dimension (str): One of 'out_features', 'channels'.
            operands (Union[LukasiewiczChannelBlock, BasePredicates]): The child logic.
            outputs_key (str): name of outputs used during explanation generation.
            add_negations (bool): add negated logic matching original logics.
        """
        super(AttentionLukasiewiczChannelOrBlock, self).__init__(
            channels, in_features, out_features, n_selected_features,
            parent_weights_dimension, operands, 'Or', outputs_key, add_negations, weight_init,
            attn_emb_dim, attn_n_layers)

    def forward(self, x: torch.Tensor):
        """
        A channel logical or.

        Args:
            x: intput tensor [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]

        Returns:
            out: output tensor [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        """
        # X: [BATCH_SIZE, 1, IN_FEATURES] OR [BATCH_SIZE, CHANNELS, 1, IN_FEATURES]
        # WEIGHTS: [CHANNELS, N_SELECTED_FEATURES, OUT_FEATURES]
        # MASK: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 3:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(1).repeat(1, self.channels, self.out_features, 1).unsqueeze(-2)
        if x.ndim == 4:
            # X AFTER TRANSPOSES: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, IN_FEATURES]
            x = x.unsqueeze(-2).repeat(1, 1, self.out_features, 1, 1)

        # X AFTER MASK: [BATCH_SIZE, CHANNELS, OUT_FEATURES, 1, N_SELECTED_FEATURES]
        x = x.gather(-1, self.mask.unsqueeze(0).expand(x.size(0), *self.mask.shape).unsqueeze(-2))

        # variational sparsity
        weights_sign_attn = self.weights.data.detach().clone().repeat(
            x.size(0), 1, 1, 1).transpose(-2, -1).unsqueeze(-2).greater_equal(torch.tensor(0.0)).float()
        # pos_attn = x * weights_sign_attn
        # neg_attn = (1 - x) * (1 - weights_sign_attn)
        # x_attn = pos_attn + neg_attn
        attn = self.sample_mask(torch.cat([x, weights_sign_attn], dim=-1))
        attn = attn.squeeze(-2).transpose(-2, -1)
        weights = self.weights.repeat(x.size(0), 1, 1, 1)

        # explicit negation
        # WEIGHTS AFTER TRANSPOSE: [CHANNELS, OUT_FEATURES, N_SELECTED_FEATURES, 1]
        weights = (weights * attn).transpose(-2, -1).unsqueeze(-1)
        weights_sign = weights.data.detach().clone().greater_equal(torch.tensor(0.0)).float()
        abs_weights = torch.abs(weights)
        pos = x @ (abs_weights * weights_sign)
        neg = (1 - x) @ (abs_weights * (1 - weights_sign))
        tot = pos + neg
        out = val_clamp(1 - self.bias + tot)

        # OUTPUT: [BATCH_SIZE, OUT_CHANNELS, 1, OUT_FEATURES]
        out = out.squeeze(-1).transpose(-2, -1)

        return out
