import numpy as np
import torch.nn as nn


class BaseReasoningNetworkMixin(nn.Module):

    def __init__(self):
        super(BaseReasoningNetworkMixin, self).__init__()
        self.model = None
        self.output_layer = None
        self.root_layer = None
        self._feature_importances = dict()

    def set_modules(
            self,
            model,
            root_layer=None
    ):
        if self.USE_DATA_PARALLEL:
            self.model = model.module.model
            self.output_layer = model.module.output_layer
            if root_layer is None:
                self.root_layer = model.module.output_layer
            else:
                self.root_layer = root_layer
        else:
            self.model = model.model
            self.output_layer = model.output_layer
            if root_layer is None:
                self.root_layer = model.output_layer
            else:
                self.root_layer = root_layer

    @staticmethod
    def _build_search_space(
            quantile: float = 0.5,
            explain_type: str = 'both',
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
    ):
        search_params = []

        # 1. rounding_precision
        # 2. ignore_uninformative
        # 3. explain_type
        # 3. ignore_uninformative and explain_type
        # 4. ignore_uninformative and rounding_precision
        # 5. ignore_uninformative and explain_type and rounding_precision
        # 6. quantile
        # 7. quantile and rounding_precision
        # 8. quantile and ignore_uninformative
        # 9. quantile and explain_type
        # 9. quantile and ignore_uninformative and explain_type
        # 9. quantile and rounding_precision and ignore_uninformative
        # 9. quantile and rounding_precision and ignore_uninformative and explain_type

        # search over rounding precisions
        for i in range(6, 15):
            search_params += [
                {'quantile': quantile,
                 'explain_type': explain_type,
                 'ignore_uninformative': ignore_uninformative,
                 'rounding_precision': i,
                 }
            ]

        # search over ignore_uninformative
        search_params += [
            {'quantile': quantile,
             'explain_type': explain_type,
             'ignore_uninformative': False,
             'rounding_precision': rounding_precision,
             }
        ]

        # search over explain_type
        search_params += [
            {'quantile': quantile,
             'explain_type': 'both',
             'ignore_uninformative': ignore_uninformative,
             'rounding_precision': rounding_precision,
             }
        ]

        # search over ignore_uninformative and explain_type
        search_params += [
            {'quantile': quantile,
             'explain_type': 'both',
             'ignore_uninformative': False,
             'rounding_precision': rounding_precision,
             }
        ]

        # search over ignore_uninformative and rounding precisions
        for i in range(6, 15):
            search_params += [
                {'quantile': quantile,
                 'explain_type': explain_type,
                 'ignore_uninformative': False,
                 'rounding_precision': i,
                 }
            ]

        # search over ignore_uninformative and explain_type and rounding precisions
        for i in range(6, 15):
            search_params += [
                {'quantile': quantile,
                 'explain_type': 'both',
                 'ignore_uninformative': False,
                 'rounding_precision': i,
                 }
            ]

        # search over quantile
        for i in np.arange(quantile, 1.0, 0.05):
            search_params += [
                {'quantile': i,
                 'explain_type': explain_type,
                 'ignore_uninformative': ignore_uninformative,
                 'rounding_precision': rounding_precision,
                 }
            ]

        # search over quantile and rounding_precision
        for i in np.arange(quantile, 1.0, 0.05):
            for j in range(6, 15):
                search_params += [
                    {'quantile': i,
                     'explain_type': explain_type,
                     'ignore_uninformative': ignore_uninformative,
                     'rounding_precision': j,
                     }
                ]

        # search over quantile and ignore_uninformative
        for i in np.arange(quantile, 1.0, 0.05):
            search_params += [
                {'quantile': i,
                 'explain_type': explain_type,
                 'ignore_uninformative': False,
                 'rounding_precision': rounding_precision,
                 }
            ]

        # search over quantile and explain_type
        for i in np.arange(quantile, 1.0, 0.05):
            search_params += [
                {'quantile': i,
                 'explain_type': 'both',
                 'ignore_uninformative': ignore_uninformative,
                 'rounding_precision': rounding_precision,
                 }
            ]

        # search over quantile and ignore_uninformative and explain_type
        for i in np.arange(quantile, 1.0, 0.05):
            search_params += [
                {'quantile': i,
                 'explain_type': 'both',
                 'ignore_uninformative': False,
                 'rounding_precision': rounding_precision,
                 }
            ]

        # search over quantile and rounding_precision and ignore_uninformative
        for i in np.arange(quantile, 1.0, 0.05):
            for j in range(6, 16):
                search_params += [
                    {'quantile': i,
                     'explain_type': explain_type,
                     'ignore_uninformative': False,
                     'rounding_precision': j,
                     }
                ]

        # search over quantile and rounding_precision and explain_type
        for i in np.arange(quantile, 1.0, 0.05):
            for j in range(6, 15):
                search_params += [
                    {'quantile': i,
                     'explain_type': 'both',
                     'ignore_uninformative': ignore_uninformative,
                     'rounding_precision': j,
                     }
                ]

        # search over quantile and rounding_precision and explain_type and ignore_uninformative
        for i in np.arange(quantile, 1.0, 0.05):
            for j in range(6, 15):
                search_params += [
                    {'quantile': i,
                     'explain_type': 'both',
                     'ignore_uninformative': False,
                     'rounding_precision': j,
                     }
                ]

        return search_params
