from collections import OrderedDict
from typing import Dict

import matplotlib.pyplot as plt
import torch

import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
)

from models import MNet, CNet
from experiments import get_train_test_functions

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, data) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        if data == 'mnist':
            # Load model
            self.model = MNet(num_classes=10)
            self.optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            self.train, self.test = get_train_test_functions(data)
        elif data == 'cifar10':
            self.model = CNet(num_classes=10)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)
            self.train, self.test = get_train_test_functions(data)

    def set_parameters(self, parameters):
        """With the model parameters received from the server,
        overwrite the uninitialise model in this class with them."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        # now replace the parameters
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract all model parameters and convert them to a list of
        NumPy arrays. The server doesn't work with PyTorch/TF/etc."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """This method train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        self.train(self.model, self.trainloader, self.optim, epochs=1)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        self.set_parameters(parameters)
        loss, accuracy = self.test(
            self.model, self.valloader
        )  # <-------------------------- calls the `test` function, just what we did in the centralised setting (but this time using the client's local validation set)
        # send statistics back to the server
        return float(loss), len(self.valloader), {"accuracy": accuracy}
