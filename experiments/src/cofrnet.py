# Imports and Seeds
import copy

from aix360.algorithms.cofrnet.CustomizedLinearClasses import CustomizedLinearFunction
from aix360.algorithms.cofrnet.CustomizedLinearClasses import CustomizedLinear
from aix360.algorithms.cofrnet.utils import generate_connections
from aix360.algorithms.cofrnet.utils import process_data
from aix360.algorithms.cofrnet.utils import train
from aix360.algorithms.cofrnet.utils import OnlyTabularDataset
from aix360.algorithms.cofrnet.CoFrNet import CoFrNet_Model
from aix360.algorithms.cofrnet.CoFrNet import generate_connections
from aix360.algorithms.cofrnet.CoFrNet import CoFrNet_Explainer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch # import main library
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
import random
from sklearn.datasets import load_breast_cancer

from sklearn.metrics import roc_auc_score


def onehot_encoding(label, n_classes):
    """Conduct one-hot encoding on a label vector."""
    label = label.view(-1)
    onehot = torch.zeros(label.size(0), n_classes).float().to(label.device)
    onehot.scatter_(1, label.view(-1, 1), 1)

    return onehot

class Cofrnet(object):

    def __init__(self, network_depth, variant, input_size, output_size, lr, momentum, epochs, weight_decay, early_stopping_plateau_count):
        self.network_depth = network_depth
        self.variant = variant
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.early_stopping_plateau_count = early_stopping_plateau_count
        self.model = CoFrNet_Model(
            generate_connections(network_depth, input_size, output_size, variant))

        self.best_model = None

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, X_train, y_train, X_holdout, y_holdout):
        # CONVERTING TO TENSOR
        tensor_x_train = torch.Tensor(X_train)
        tensor_y_train = torch.Tensor(y_train).long()

        tensor_x_holdout = torch.Tensor(X_holdout)
        tensor_y_holdout = torch.Tensor(y_holdout).long()

        train_dataset = OnlyTabularDataset(tensor_x_train,
                                           tensor_y_train)

        batch_size = 128
        dataloader = DataLoader(train_dataset, batch_size)
        self._train(
            self.model,
            dataloader,
            tensor_x_holdout,
            tensor_y_holdout,
            self.output_size,
            epochs=self.epochs,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,

        )

    def predict(self, X):
        tensor_x = torch.Tensor(X)
        if torch.cuda.is_available():
            tensor_x = tensor_x.cuda()
        return self.model(tensor_x).detach().cpu()

    def eval(self, X, y):
        if y.shape[1] == 1:
            predictions = self.predict(X)[:, -1]
        else:
            predictions = self.predict(X)
        predictions = predictions.cpu()
        return roc_auc_score(y, predictions, multi_class='ovo', average='micro')

    def _train(self,
               model,
               dataloader,
               X_holdout,
               y_holdout,
               num_classes,
               lr=0.001,
               momentum=0.9,
               epochs=20,
               weight_decay=0.00001
               ):
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss(reduction="sum")
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(momentum, 0.999))

        best_holdout_score = 0.0
        plateau_count = 0

        EPOCHS = epochs
        for epoch in range(EPOCHS):  # loop over the dataset multiple times
            print("Epoch: ", epoch)
            if epoch > 0:
                holdout_score = self.eval(X_holdout, y_holdout)
                if holdout_score > best_holdout_score:
                    print(f"Best holdout score updated from {best_holdout_score} to {holdout_score}")
                    best_holdout_score = holdout_score
                    self.best_model = copy.deepcopy(model)
                    plateau_count = 0
                else:
                    plateau_count += 1
                if plateau_count >= self.early_stopping_plateau_count:
                    break

            running_loss = 0.0
            # for i, data in enumerate(trainloader, 0):
            for i, batch in tqdm(enumerate(dataloader)):
                # get the inputs; data is a list of [inputs, labels]
                # forward + backward + optimize

                tabular = batch['tabular'].cuda()
                target = batch['target'].cuda()

                tabular.requires_grad = True
                if torch.cuda.is_available():
                    tabular = tabular.cuda()
                    target = target.cuda()

                outputs = model(tabular)

                one_hot_encoded_target = onehot_encoding(target, num_classes)

                # loss = criterion(outputs, batch['target'])
                loss = criterion(outputs, one_hot_encoded_target)

                # zero the parameter gradients
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            print("Loss: ", running_loss)

        self.model = self.best_model

        print('Finished Training')