from typing import Callable, Union
import numpy as np
import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class IdToSizeFncPartitioner(Partitioner):
    def __init__(
        self,
        num_partitions: int,
        partition_id_to_size_fn: Callable, 
    ) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions
        self._partition_id_to_size_fn = partition_id_to_size_fn

        self._partition_id_to_size: dict[int, int] = {}
        self._partition_id_to_indices: dict[int, list[int]] = {}

        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    @property
    def partition_id_to_size(self) -> dict[int, int]:
        return self._partition_id_to_size

    @property
    def partition_id_to_indices(self) -> dict[int, list[int]]:
        return self._partition_id_to_indices

    def _determine_partition_id_to_size(self) -> None:
        data_division_in_units = self._partition_id_to_size_fn(
            np.linspace(start=1, stop=self._num_partitions, num=self._num_partitions)
        )
        total_units: Union[int, float] = data_division_in_units.sum()

        partition_sizes_as_fraction = data_division_in_units / total_units

        partition_sizes_as_num_of_samples = np.array(
            partition_sizes_as_fraction * len(self.dataset), dtype=np.int64
        )

        assigned_samples = np.sum(partition_sizes_as_num_of_samples)
        left_unassigned_samples = len(self.dataset) - assigned_samples

        partition_sizes_as_num_of_samples[-1] += left_unassigned_samples
        for idx, partition_size in enumerate(partition_sizes_as_num_of_samples):
            self._partition_id_to_size[idx] = partition_size

        self._check_if_partition_id_to_size_possible()

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        if self._partition_id_to_indices_determined:
            return
        self._determine_partition_id_to_size()
        total_samples_assigned = 0
        for idx, quantity in self._partition_id_to_size.items():
            self._partition_id_to_indices[idx] = list(
                range(total_samples_assigned, total_samples_assigned + quantity)
            )
            total_samples_assigned += quantity
        self._partition_id_to_indices_determined = True

    def _check_if_partition_id_to_size_possible(self) -> None:
        all_positive = all(value >= 1 for value in self.partition_id_to_size.values())
        if not all_positive:
            raise ValueError(
                f"The given specification of the parameter num_partitions"
                f"={self._num_partitions} for the given dataset results "
                f"in the partitions sizes that are not greater than 0."
            )


class ExponentialPartitioner(IdToSizeFncPartitioner):
    def __init__(self, num_partitions: int, total_samples: int, growth_factor: float = 2.0, min_samples_per_partition: int = 10) -> None:
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        if total_samples < num_partitions * min_samples_per_partition:
            raise ValueError("Total samples are too few to allocate the minimum required samples per partition.")

        self._num_partitions = num_partitions  # Use _num_partitions to avoid conflict
        self._total_samples = total_samples
        self._growth_factor = growth_factor
        self._min_samples_per_partition = min_samples_per_partition

        def partition_id_to_size_fn(partition_id: int) -> int:
            partition_id = int(partition_id)
            
            remaining_samples = total_samples - (num_partitions * min_samples_per_partition)

            exponents = np.array([growth_factor ** i for i in range(num_partitions)])

            normalized_values = (exponents / exponents.sum()) * remaining_samples

            sample_sizes = np.round(normalized_values).astype(int) + min_samples_per_partition

            return sample_sizes[partition_id]

        # Call the superclass constructor with the exponential partition function
        super().__init__(
            num_partitions=num_partitions,
            partition_id_to_size_fn=partition_id_to_size_fn
        )

    def _determine_partition_id_to_size(self) -> None:
        """Determine data quantity associated with partition indices."""
        # Generate exponential distribution values
        exponents = np.array([self._growth_factor ** i for i in range(self._num_partitions)])

        remaining_samples = self._total_samples - (self._num_partitions * self._min_samples_per_partition)

        normalized_values = (exponents / exponents.sum()) * remaining_samples

        partition_sizes_as_num_of_samples = np.round(normalized_values).astype(int) + self._min_samples_per_partition

        assigned_samples = np.sum(partition_sizes_as_num_of_samples)
        left_unassigned_samples = self._total_samples - assigned_samples

        while left_unassigned_samples > 0:
            for idx in np.argsort(partition_sizes_as_num_of_samples)[::-1]:
                if left_unassigned_samples > 0:
                    partition_sizes_as_num_of_samples[idx] += 1
                    left_unassigned_samples -= 1

        for idx, partition_size in enumerate(partition_sizes_as_num_of_samples):
            self._partition_id_to_size[idx] = partition_size

        self._check_if_partition_id_to_size_possible()

