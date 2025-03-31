# main.py

import time
import numpy as np
import flwr as fl
import pickle
from typing import List, Dict, Any
from utils import load_config, override_config, set_seed, load_datasets, generate_client_fn
from strategy import get_strategies

def run_experiment(config):
    num_clients = config["num_clients"]
    num_rounds = config["num_rounds"]
    n_iterations = config["n_iterations"]

    results = {}

    for partition in config["partitions"]:
        results[partition] = {}
        for data in config["datasets"]:
            results[partition][data] = {}
            trainloaders, valloaders, testloader = load_datasets(data, config["val_split"], partition, config)
            client_fn_callback = generate_client_fn(trainloaders, valloaders, data=data)

            strategies = get_strategies(testloader, data)

            for label, strategy in strategies.items():
                accuracies = []
                losses_c = []
                losses_d = []
                times = []

                for _ in range(n_iterations):
                    start_time = time.time()

                    history = fl.simulation.start_simulation(
                        client_fn=client_fn_callback,
                        num_clients=num_clients,
                        config=fl.server.ServerConfig(num_rounds=num_rounds),
                        strategy=strategy,
                    )

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    accuracy = [data[1] * 100 for data in history.metrics_centralized["accuracy"]]
                    loss_c = [data[1] for data in history.losses_centralized]
                    loss_d = [data[1] for data in history.losses_distributed]

                    accuracies.append(accuracy)
                    losses_c.append(loss_c)
                    losses_d.append(loss_d)
                    times.append(elapsed_time)

                results[partition][data][label] = {
                    'accuracies': accuracies,
                    'losses_distributed': losses_d,
                    'losses_centralized': losses_c,
                    'times': times
                }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--nclients", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--partition", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    config = override_config(config, args)
    set_seed(config.get("seed", 42))

    results = run_experiment(config)
    pickle_path = config.get("result_path", "results_global.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()