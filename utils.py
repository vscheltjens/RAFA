import random
import pickle
import yaml
from collections import OrderedDict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, LinearPartitioner, IidPartitioner

from typing import List, Dict, Any
from models import MNet, CNet
from client import FlowerClient
from experiments import get_train_test_functions
from partitioners import ExponentialPartitioner


def get_evaluate_fn(testloader, data):
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""

    def evaluate_fn(server_round: int, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        if data == 'mnist':
            # Load model
            model = MNet(num_classes=10)
            _, test = get_train_test_functions(data)
        elif data == 'cifar10':
            model = CNet(num_classes=10)
            _, test = get_train_test_functions(data)

        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # call test
        loss, accuracy = test(
            model, testloader
        )  # <-------------------------- calls the `test` function, just what we did in the centralised setting
        return loss, {"accuracy": accuracy}

    return evaluate_fn

def generate_client_fn(trainloaders, valloaders, data):
    def client_fn(cid: str):
        """Returns a FlowerClient containing the cid-th data partition"""

        return FlowerClient(
            trainloader=trainloaders[int(cid)], valloader=valloaders[int(cid)],
        data=data).to_client()

    return client_fn


def save_results_to_pickle(results: Dict[str, Any], filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def visualize_data_distribution_side_by_side(trainloaders_dict):
    """Visualize the distribution of the number of samples per client for multiple partition schemes side by side."""
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # 2 rows, 2 columns for 4 partition types
    partition_types = ['iid', 'linear', 'exponential', 'dirichlet']
    
    # Iterate over partition types and plot in a 2x2 grid
    for i, partition in enumerate(partition_types):
        trainloaders = trainloaders_dict[partition]
        client_sample_counts = {i: len(loader.dataset) for i, loader in enumerate(trainloaders)}
        sorted_client_sample_counts = dict(sorted(client_sample_counts.items(), key=lambda item: item[1], reverse=True))
        
        # Calculate the position in the 2x2 grid
        row, col = divmod(i, 2)
        
        axs[row, col].bar(range(len(sorted_client_sample_counts)), list(sorted_client_sample_counts.values()), color='purple', alpha=0.5)
        axs[row, col].grid()
        axs[row, col].set_xlabel("Client index")
        axs[row, col].set_ylabel("Number of samples")
        axs[row, col].set_title(f"{partition.capitalize()} partition")
    
    plt.tight_layout()
    plt.savefig('dist_all_partitions_2x2.pdf', bbox_inches='tight')
    plt.show()


def plot_results(results):
    scenarios = ['dirichlet', 'linear', 'iid', 'exponential']
    datasets = ['mnist', 'cifar10']
    strategies = ['FedAvg', 'FedProx', 'FedAvgM', 'RAFA']

    # Create a 2x4 grid for plotting
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    # Loop through each scenario and dataset to plot the results
    for i, scenario in enumerate(scenarios):
        for j, dataset in enumerate(datasets):
            ax = axs[j, i]
            
            # Plot accuracies for each strategy
            for strategy in strategies:
                # Getting accuracies for two iterations and computing the mean and standard deviation
                accuracies = np.array(results[scenario][dataset][strategy]['accuracies'])
                mean_accuracies = np.mean(accuracies, axis=0)
                std_accuracies = np.std(accuracies, axis=0)
                
                # Plot the mean accuracy with standard deviation as shaded area
                ax.plot(mean_accuracies, label=strategy)
                ax.fill_between(range(len(mean_accuracies)), 
                                mean_accuracies - std_accuracies, 
                                mean_accuracies + std_accuracies, 
                                alpha=0.2)

            # Set title and legend
            ax.set_title(f'{scenario.capitalize()} - {dataset.upper()}')
            ax.legend()
            ax.set_xlabel('Rounds')
            ax.set_ylabel('Accuracy')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def override_config(config: dict, args) -> dict:
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None:
            config[key] = value
    return config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets(dataset, alpha, part, config):

    if part == 'linear':
        partitioner = LinearPartitioner(num_partitions=config["num_clients"])
    elif part == 'exponential':
        if dataset == 'mnist':
            partitioner = ExponentialPartitioner(100, 59998, growth_factor=1.4, min_samples_per_partition=200)
        else:
            partitioner = ExponentialPartitioner(100, 49998, growth_factor=1.4, min_samples_per_partition=200)
    elif part == 'iid':
        partitioner = IidPartitioner(num_partitions=config["num_clients"])
    elif part == 'dirichlet':
        partitioner = DirichletPartitioner(
            num_partitions=config["num_clients"], 
            partition_by="label",
            alpha=alpha, 
            min_partition_size=10,
            self_balancing=False
        )

    fds = FederatedDataset(dataset=dataset, partitioners={"train": partitioner})

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        if dataset == 'cifar10':
        
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            batch["img"] = [transform(img) for img in batch["img"]]

        elif dataset == 'mnist':
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        
            # Identify the correct key for images and rename it to 'img'
            image_key = 'img' if 'img' in batch else 'image'
            batch['img'] = [transform(img) for img in batch.pop(image_key)]
                
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for partition_id in range(config["num_clients"]):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8, seed=42)
        trainloaders.append(DataLoader(partition["train"], batch_size=config["batch_size"]))
        valloaders.append(DataLoader(partition["test"], batch_size=config["batch_size"]))
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=config["batch_size"])
    return trainloaders, valloaders, testloader


