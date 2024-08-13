import torch
from torchlogic.modules import BanditNRNModule


class TestBaseClassifier:

    @staticmethod
    def test__init():

        layer_sizes = [5, 5]
        input_size=4
        output_size = 2

        module = BanditNRNModule(
            input_size,
            output_size,
            layer_sizes,
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            n_selected_features_input=2,
            n_selected_features_internal=1,
            n_selected_features_output=1,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        assert len(module.model) == len(layer_sizes)
        assert module.model[0].in_features == input_size
        assert all([m.channels == output_size for m in module.model])
        assert module.output_layer.channels == output_size

    @staticmethod
    def test__forward():
        layer_sizes = [5, 5]
        input_size = 4
        output_size = 2

        module = BanditNRNModule(
            input_size,
            output_size,
            layer_sizes,
            feature_names=['feat1', 'feat1', 'feat3', 'feat4'],
            n_selected_features_input=2,
            n_selected_features_internal=1,
            n_selected_features_output=1,
            perform_prune_quantile=0.5,
            ucb_scale=2.5
        )

        x = torch.ones(10, 4)
        output = module(x)

        assert output.size() == (x.size(0), output_size)
        assert output.min() >= 0
        assert output.max() <= 1


