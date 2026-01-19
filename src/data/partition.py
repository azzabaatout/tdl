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
        elif self.partition_type == 'non_iid':
            return self._non_iid_partition()
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

    def _non_iid_partition(self):
        """
        Non-IID partition using the 'c' parameter method from the paper.

        alpha = number of classes each client can have (c in the paper)
        - alpha=10: equivalent to IID (all classes)
        - alpha=5: each client gets data from 5 random classes
        - alpha=2: highly non-IID (each client gets data from 2 classes)
        """
        c = int(self.alpha)
        c = min(c, self.num_classes)

        # group indices by class
        class_indices = {i: np.where(self.targets == i)[0] for i in range(self.num_classes)}

        # shuffle indices within each class
        for cls in class_indices:
            np.random.shuffle(class_indices[cls])

        # track how much of each class has been assigned
        class_pointers = {i: 0 for i in range(self.num_classes)}

        client_data = []
        samples_per_client = len(self.dataset) // self.num_clients

        for client_id in range(self.num_clients):
            # randoml select c classes for this client
            client_classes = np.random.choice(self.num_classes, c, replace=False)

            # collect samples from selected classes
            client_indices = []
            samples_per_class = samples_per_client // c

            for cls in client_classes:
                start = class_pointers[cls]
                end = start + samples_per_class

                # wrap around if we run out of samples in this class
                available = class_indices[cls]
                if end > len(available):
                    # reset and reshuffle
                    np.random.shuffle(class_indices[cls])
                    class_pointers[cls] = 0
                    start = 0
                    end = samples_per_class

                indices = class_indices[cls][start:end]
                client_indices.extend(indices)
                class_pointers[cls] = end

            client_data.append(Subset(self.dataset, client_indices))

        return client_data

