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
        elif self.partition_type == 'non_iid_dirichlet':
            return self._non_iid_dirichlet_partition()
        elif self.partition_type == 'non_iid_class':
            return self._non_iid_class_partition()
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
    
    def _non_iid_dirichlet_partition(self):
        min_size = 0
        K = self.num_classes
        N = len(self.dataset)
        
        while min_size < 10:
            idx_batch = [[] for _ in range(self.num_clients)]
            for k in range(K):
                idx_k = np.where(self.targets == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
                proportions = np.array([p * (len(idx_j) < N / self.num_clients) 
                                      for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                idx_batch = [idx_j + idx.tolist() 
                           for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        
        client_data = []
        for indices in idx_batch:
            client_data.append(Subset(self.dataset, indices))
            
        return client_data
    
    def _non_iid_class_partition(self, classes_per_client=2):
        num_shards = self.num_clients * classes_per_client
        shard_size = len(self.dataset) // num_shards
        
        sorted_indices = np.argsort(self.targets)
        shard_indices = []
        
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < num_shards - 1 else len(self.dataset)
            shard_indices.append(sorted_indices[start_idx:end_idx])
        
        np.random.shuffle(shard_indices)
        
        client_data = []
        for i in range(self.num_clients):
            client_indices = []
            for j in range(classes_per_client):
                shard_idx = i * classes_per_client + j
                client_indices.extend(shard_indices[shard_idx])
            client_data.append(Subset(self.dataset, client_indices))
            
        return client_data