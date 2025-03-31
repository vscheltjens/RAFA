import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from logging import DEBUG, INFO, WARNING
from sklearn.metrics.pairwise import cosine_similarity

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from utils import get_evaluate_fn

class RAFA(Strategy):
    """Federated Averaging strategy with refined client selection and weighted averaging."""

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        adaptive_alpha: float = 0.5,
        adaptive_beta: float = 0.5,
        adaptive_gamma: float = 0.5,
        top_fraction: float = 0.1,
        learning_rate: float = 0.5
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, "Minimum number of available clients is too low")

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.historical_contributions: Dict[str, List[float]] = {}
        self.client_updates: Dict[str, List[NDArrays]] = {}
        self.selected_clients = None
        self.alpha = adaptive_alpha
        self.beta = adaptive_beta
        self.gamma = adaptive_gamma
        self.top_fraction = top_fraction
        self.global_parameters: Optional[Parameters] = initial_parameters
        self.prev_performance = None
        self.learning_rate = learning_rate
        self.main_contribution_scores: Dict[str, float] = {}  # To log main contribution scores

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"RAFA(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        if self.initial_parameters is not None:
            self.global_parameters = self.initial_parameters
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def aggregate_metrics(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate client-level metrics to round-level metrics."""
        aggregated_metrics = {}
        for client_id, client_metrics in metrics.items():
            for metric, value in client_metrics.items():
                if metric not in aggregated_metrics:
                    aggregated_metrics[metric] = 0.0
                aggregated_metrics[metric] += value
        # Average the aggregated metrics
        num_clients = len(metrics)
        for metric in aggregated_metrics:
            aggregated_metrics[metric] /= num_clients
        return aggregated_metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res

        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = (parameters, config)

        # Sample clients
        if server_round == 1:
            # In the first round, use a fraction of the clients
            sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
            clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
            self.selected_clients = [client.cid for client in clients]
            return [(client, fit_ins) for client in clients]

        # Select top clients and add some randomness
        top_clients = self.select_top_clients(client_manager)
        random_clients = self.select_random_clients(client_manager, 5)
        selected_clients = top_clients + random_clients

        self.selected_clients = [client.cid for client in selected_clients]
        log(INFO, f"Round {server_round}: Selected clients {self.selected_clients}")

        return [(client, fit_ins) for client in selected_clients]

    def update_contribution_scores(self, results: List[Tuple[ClientProxy, FitRes]], global_parameters: Parameters) -> None:
        global_params_ndarrays = parameters_to_ndarrays(global_parameters)
        current_performance = {}
        for client, fit_res in results:
            client_id = client.cid
            hcs = self.evaluate_historical_contribution(client_id)
            siu = self.evaluate_significance_of_update(fit_res)
            iu = self.evaluate_informativeness_of_update(fit_res, global_params_ndarrays)
            ccs = self.alpha * hcs + self.beta * siu + self.gamma * iu
            current_performance[client_id] = {'historical_contribution': hcs, 'significance_update': siu, 'informativeness_update': iu}
            self.main_contribution_scores[client_id] = ccs  # Log the main contribution score
            if client_id in self.historical_contributions:
                self.historical_contributions[client_id].append(ccs)
                self.client_updates[client_id].append(fit_res.parameters)
            else:
                self.historical_contributions[client_id] = [ccs]
                self.client_updates[client_id] = [fit_res.parameters]

            if self.prev_performance is not None:
                current_aggregated_metrics = self.aggregate_metrics(current_performance)
                prev_aggregated_metrics = self.aggregate_metrics(self.prev_performance)
                self.update_adaptive_weights(current_aggregated_metrics, prev_aggregated_metrics)
            self.prev_performance = current_performance


    def evaluate_historical_contribution(self, client_id: str) -> float:
        if client_id in self.historical_contributions:
            return np.mean(self.historical_contributions[client_id])
        return 0.0

    def evaluate_significance_of_update(self, fit_res: FitRes) -> float:
        # Significance can be evaluated as the norm of the parameter updates
        updates = parameters_to_ndarrays(fit_res.parameters)
        return np.mean([np.linalg.norm(update) for update in updates])

    def evaluate_informativeness_of_update(self, fit_res: FitRes, global_params: NDArrays) -> float:
        # Informativeness can be evaluated using cosine similarity between the update and the global model
        updates = parameters_to_ndarrays(fit_res.parameters)
        return np.mean([cosine_similarity(update.flatten().reshape(1, -1), global_params[layer_idx].flatten().reshape(1, -1))[0, 0] for layer_idx, update in enumerate(updates)])

    def recency_weighted_average(self, contributions: List[float], decay_factor: float = 0.9) -> float:
        """Calculate the recency-weighted average of contributions."""
        weights = [decay_factor ** i for i in range(len(contributions))]
        weighted_sum = sum(w * c for w, c in zip(weights, reversed(contributions)))
        return weighted_sum / sum(weights)

    def calculate_entropy(self, updates: List[Parameters]) -> float:
        """Calculate the entropy of the client's updates."""
        flat_updates = np.concatenate([np.concatenate(parameters_to_ndarrays(update)).flatten() for update in updates])
        probabilities, _ = np.histogram(flat_updates, bins=256, density=True)
        probabilities = probabilities[probabilities > 0]  # Remove zero entries
        entropy = -np.sum(probabilities * np.log(probabilities))
        return entropy

    def performance_loss(self, current_performance: Dict[str, float], prev_performance: Dict[str, float]) -> float:
        loss = 0.0
        for metric in current_performance:
            if metric in prev_performance:
                loss += (current_performance[metric] - prev_performance[metric]) ** 2
        return loss

    def select_top_clients(self, client_manager: ClientManager) -> List[ClientProxy]:
        """Select top 5 clients based on recency-weighted contributions and entropy."""
        if not self.historical_contributions:
            return []

        # Calculate recency-weighted average contributions and entropy
        recency_weighted_contributions = {
            cid: self.recency_weighted_average(scores) for cid, scores in self.historical_contributions.items()
        }
        # entropies = {
        #     cid: self.calculate_entropy(self.client_updates[cid]) for cid in self.historical_contributions.keys()
        # }

        # Combine the metrics to score each client
        # combined_scores = {
        #     cid: recency_weighted_contributions[cid] + entropies[cid]
        #     for cid in recency_weighted_contributions
        # }

        combined_scores = {
            cid: recency_weighted_contributions[cid]
            for cid in recency_weighted_contributions
        }

        # Select top 5 clients based on combined scores
        sorted_clients = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        top_client_ids = [cid for cid, _ in sorted_clients[:5]]
        all_clients = client_manager.all()
        top_clients = [all_clients[cid] for cid in top_client_ids if cid in all_clients]

        return top_clients

    def select_random_clients(self, client_manager: ClientManager, num_random_clients: int) -> List[ClientProxy]:
        """Select random clients to ensure diversity."""
        remaining_clients = [client for client in client_manager.all().values() if client.cid not in self.selected_clients]
        random_clients = random.sample(remaining_clients, num_random_clients)
        return random_clients

    def update_adaptive_weights(self, current_performance: Dict[str, Scalar], prev_performance: Dict[str, Scalar]) -> None:
        """Update the adaptive weights \alpha, \beta, and \gamma based on performance improvement."""
        if not prev_performance:
            return

        loss = self.performance_loss(current_performance, prev_performance)

        # Compute the gradients (simplified for illustration)
        grad_alpha = 2 * (current_performance.get("historical_contribution", 0) - prev_performance.get("historical_contribution", 0))
        grad_beta = 2 * (current_performance.get("significance_update", 0) - prev_performance.get("significance_update", 0))
        grad_gamma = 2 * (current_performance.get("informativeness_update", 0) - prev_performance.get("informativeness_update", 0))

        # Update the weights using gradient descent
        self.alpha -= self.learning_rate * grad_alpha
        self.beta -= self.learning_rate * grad_beta
        self.gamma -= self.learning_rate * grad_gamma

        # Enforce a minimum contribution of 15% for each parameter
        min_contribution = 0.15
        total_weight = self.alpha + self.beta + self.gamma

        self.alpha = max(self.alpha, min_contribution * total_weight)
        self.beta = max(self.beta, min_contribution * total_weight)
        self.gamma = max(self.gamma, min_contribution * total_weight)

        # Normalize weights to sum to 1
        total_weight = self.alpha + self.beta + self.gamma
        self.alpha /= total_weight
        self.beta /= total_weight
        self.gamma /= total_weight

        log(INFO, f'Alpha {self.alpha}, Beta {self.beta}, Gamma {self.gamma}')

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average based on contribution score."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Extract weights and contributions
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Ensure global parameters are set
        if self.global_parameters is None:
            log(INFO, "Setting initial global parameters")
            self.global_parameters = ndarrays_to_parameters(weights_results[0][0])

        global_params_ndarrays = parameters_to_ndarrays(self.global_parameters)
        contributions = [
            self.evaluate_contribution(client, fit_res, global_params_ndarrays)
            for client, fit_res in results
        ]
        total_contributions = sum(contributions)

        # Initialize aggregated updates
        aggregated_updates = [np.zeros_like(layer) for layer in weights_results[0][0]]

        for (client_updates, num_examples), contribution in zip(weights_results, contributions):
            for layer_idx, update in enumerate(client_updates):
                aggregated_updates[layer_idx] += update * (contribution / total_contributions)

        parameters_aggregated = ndarrays_to_parameters(aggregated_updates)

        # Update global parameters
        self.global_parameters = parameters_aggregated

        # Update contribution scores after aggregation
        self.update_contribution_scores(results, parameters_aggregated)

        return parameters_aggregated, {}

    def evaluate_contribution(self, client: ClientProxy, fit_res: FitRes, global_params: NDArrays) -> float:
        hcs = self.evaluate_historical_contribution(client.cid)
        siu = self.evaluate_significance_of_update(fit_res)
        iu = self.evaluate_informativeness_of_update(fit_res, global_params)
        return self.alpha * hcs + self.beta * siu + self.gamma * iu

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        return loss_aggregated, metrics_aggregated


def get_strategies(testloader, dataset_name):
    return {
        "FedAvg": fl.server.strategy.FedAvg(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_available_clients=100,
            evaluate_fn=get_evaluate_fn(testloader, dataset_name),
        ),
        "RAFA": RAFA(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_available_clients=100,
            evaluate_fn=get_evaluate_fn(testloader, dataset_name),
            min_fit_clients=2,
            min_evaluate_clients=2,
        ),
        "FedProx": fl.server.strategy.FedProx(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_available_clients=100,
            evaluate_fn=get_evaluate_fn(testloader, dataset_name),
            proximal_mu=0.1,
        ),
        "FedAvgM": fl.server.strategy.FedAvgM(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_available_clients=100,
            evaluate_fn=get_evaluate_fn(testloader, dataset_name),
            min_fit_clients=2,
            min_evaluate_clients=2,
        ),
    }