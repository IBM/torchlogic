import torch
import numpy as np
from torch.utils.data import Dataset


class SimpleDataset(Dataset):

    """
    Class for a simple dataset used in sklogic.
    """

    def __init__(
            self,
            X: np.array,
            y: np.array
    ):
        """
        Dataset suitable for SKLogic models model from torchlogic

        Args:
            X (np.array): features data scaled to [0, 1]
            y (np.array): target data of classes 0, 1
        """
        super(SimpleDataset, self).__init__()
        self.X = X.astype('float')
        self.y = y
        self.sample_idx = np.arange(X.shape[0])  # index of samples

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        features = torch.from_numpy(self.X[idx, :]).float()
        target = torch.from_numpy(self.y[idx, :])
        return {'features': features, 'target': target, 'sample_idx': idx}