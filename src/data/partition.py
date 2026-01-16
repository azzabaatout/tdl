import numpy as np
from torch.utils.data import Subset


class DataPartitioner:
    def __init__(self, dataset, num_clients, partition_type='iid', alpha=0.5):
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition_type = partition_type
        self.alpha = alpha
        self.targets = np.array([dataset[i][1] for i in range(len(dataset))])
        self.num_classes = len(np.unique(self.targets))

    def partition_data(self):
        if self.partition_type == 'iid':
            return self._iid_partition()
        else:
            raise ValueError(f"Unknown partition type: {self.partition_type}")

    def _iid_partition(self):
        num_samples = len(self.dataset) // self.num_clients
        indices = np.random.permutation(len(self.dataset))

        client_data = []
        for i in range(self.num_clients):
            start_idx = i * num_samples
            end_idx = start_idx + num_samples if i < self.num_clients - 1 else len(self.dataset)
            client_indices = indices[start_idx:end_idx]
            client_data.append(Subset(self.dataset, client_indices))

        return client_data

