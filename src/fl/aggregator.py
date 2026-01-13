import torch
from collections import OrderedDict

from src.models.model_utils import flatten_model


class Aggregator:
    def __init__(self, aggregation_method='fedavg'):
        self.aggregation_method = aggregation_method
    
    def aggregate(self, client_models, client_weights=None):
        """agg client models into a global model."""
        if self.aggregation_method == 'fedavg':
            return self._federated_averaging(client_models, client_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _federated_averaging(self, client_models, client_weights):
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # weights normalization
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # agg aprams
        global_state_dict = OrderedDict()
        
        # get the structure from the first model
        for name, param in client_models[0].named_parameters():
            global_state_dict[name] = torch.zeros_like(param)
        
        # weighted sum of all client models
        for i, model in enumerate(client_models):
            for name, param in model.named_parameters():
                global_state_dict[name] += client_weights[i] * param.data
        
        return global_state_dict


class DefensiveAggregator(Aggregator):
    def __init__(self, defense_method='none', **defense_params):
        super().__init__()
        self.defense_method = defense_method
        self.defense_params = defense_params

    def aggregate(self, client_models, client_weights=None, server_model=None, client_ids=None, current_round=None):
        """Aggregate with defensive mechanisms."""
        if self.defense_method == 'none':
            return self._federated_averaging(client_models, client_weights)
        elif self.defense_method == 'krum':
            return self._krum_aggregation(client_models, client_ids)
        elif self.defense_method == 'norm_clipping':
            return self._norm_clipping_aggregation(client_models, client_weights, client_ids)
        elif self.defense_method == 'fltrust':
            return self._fltrust_aggregation(client_models, server_model, client_ids)
        elif self.defense_method in ['spp', 'adaptive_spp']:
            return self._spp_aggregation(client_models, server_model, client_ids, current_round)
        else:
            raise ValueError(f"Unknown defense method: {self.defense_method}")
    
    def _krum_aggregation(self, client_models):
        """Krum defense: select one model with minimum distance sum"""
        n_models = len(client_models)
        m = self.defense_params.get('num_malicious', n_models // 4)
        
        # calc pairwise distances
        distances = torch.zeros(n_models, n_models)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                flat_i = flatten_model(client_models[i])
                flat_j = flatten_model(client_models[j])
                dist = torch.norm(flat_i - flat_j).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        # calc scores for each model
        scores = []
        for i in range(n_models):
            # sort distances for model i and sum the smallest n-m-1 distances
            sorted_distances = torch.sort(distances[i])[0]
            score = torch.sum(sorted_distances[1:n_models-m]).item()  # Exclude self (distance 0)
            scores.append(score)
        
        # select model with minimum score
        selected_idx = scores.index(min(scores))
        selected_model = client_models[selected_idx]
        
        # return the selected model's state dictionnary
        return OrderedDict(selected_model.named_parameters())
    
    def _norm_clipping_aggregation(self, client_models, client_weights):
        """Norm clipping defense: clip models exceeding threshold"""
        max_norm = self.defense_params.get('max_norm', 10.0)
        
        filtered_models = []
        filtered_weights = []
        
        for i, model in enumerate(client_models):
            model_norm = 0
            for param in model.parameters():
                model_norm += torch.norm(param.data) ** 2
            model_norm = model_norm ** 0.5
            
            if model_norm <= max_norm:
                filtered_models.append(model)
                if client_weights:
                    filtered_weights.append(client_weights[i])
        
        if not filtered_models:
            # If all models are rejected, use the first one
            filtered_models = [client_models[0]]
            filtered_weights = [1.0]
        
        return self._federated_averaging(filtered_models, filtered_weights)
    
    def _fltrust_aggregation(self, client_models, server_model):
        """FLTrust defense: use server model as reference"""
        if server_model is None:
            raise ValueError("Server model required for FLTrust defense")
        
        # calc cosine similarities with server model
        similarities = []
        for model in client_models:
            sim = self._cosine_similarity(model, server_model)
            similarities.append(max(0, sim))  # ReLU activation
        
        # normalize similarities as weights
        total_sim = sum(similarities)
        if total_sim == 0:
            weights = [1.0 / len(client_models)] * len(client_models)
        else:
            weights = [sim / total_sim for sim in similarities]
        
        return self._federated_averaging(client_models, weights)
    
    def _cosine_similarity(self, model1, model2):
        flat1 = flatten_model(model1)
        flat2 = flatten_model(model2)
        
        dot_product = torch.dot(flat1, flat2)
        norm1 = torch.norm(flat1)
        norm2 = torch.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()