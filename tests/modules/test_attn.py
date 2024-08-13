import torch
from torchlogic.modules import AttentionNRNModule
from torchlogic.nn import (ConcatenateBlocksLogic, AttentionLukasiewiczChannelAndBlock,
                           AttentionLukasiewiczChannelOrBlock)


class TestAttnNRNModule:

    @staticmethod
    def test__init():

        layer_sizes = [5, 5]
        input_size=4
        output_size = 2

        module = AttentionNRNModule(
            input_size,
            output_size,
            layer_sizes,
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            add_negations=False,
            weight_init=0.2,
            attn_emb_dim=8,
            attn_n_layers=2,
            normal_form='dnf'
        )

        for i in range(len(module.model)):
            if i % 2 == 0:
                assert isinstance(module.model[i], AttentionLukasiewiczChannelAndBlock), "incorrect block type"
            else:
                assert isinstance(module.model[i], AttentionLukasiewiczChannelOrBlock), "incorrect block type"
            assert module.model[i].channels == output_size, "channels is incorrect"
        assert isinstance(module.output_layer, AttentionLukasiewiczChannelAndBlock), \
            "output_layer should be of type 'AttentionLukasiewiczChannelAndBlock'"
        assert len(module.model) == len(layer_sizes), "number of layers is incorrect"
        assert module.model[0].in_features == input_size, \
            "disjuction input layer's input size is incorrect"
        assert module.output_layer.channels == output_size

        module = AttentionNRNModule(
            input_size,
            output_size,
            layer_sizes,
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            add_negations=True,
            weight_init=0.2,
            attn_emb_dim=8,
            attn_n_layers=2,
            normal_form='cnf'
        )

        for i in range(len(module.model)):
            if i % 2 == 0:
                assert isinstance(module.model[i], AttentionLukasiewiczChannelOrBlock), "incorrect block type"
            else:
                assert isinstance(module.model[i], AttentionLukasiewiczChannelAndBlock), "incorrect block type"
            assert module.model[i].channels == output_size, "channels is incorrect"
        assert isinstance(module.output_layer, AttentionLukasiewiczChannelOrBlock), \
            "output_layer should be of type 'AttentionLukasiewiczChannelOrBlock'"
        assert module.output_layer.channels == output_size

    @staticmethod
    def test__forward():
        layer_sizes = [5, 5]
        input_size = 4
        output_size = 2

        module = AttentionNRNModule(
            input_size,
            output_size,
            layer_sizes,
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
        )

        x = torch.ones(10, 4)
        output = module(x)

        assert output.size() == (x.size(0), output_size)
        assert output.min() >= 0
        assert output.max() <= 1


