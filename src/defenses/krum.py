import torch
import numpy as np
from .base_defense import SimilarityBasedDefense


class Krum(SimilarityBasedDefense):
    """
    Krum Defense: Select one model with minimum distance sum to other models.
    Assumes knowledge of the number of malicious clients.
    """
    
    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.num_malicious = defense_params.get('num_malicious', 1)
        self.multi_krum = defense_params.get('multi_krum', False)
        self.num_selected = defense_params.get('num_selected', 1)
    
    def defend(self, client_models, **kwargs):
        """
        Apply Krum defense mechanism.
        
        Args:
            client_models: List of client models
        
        Returns:
            dict: Defense results with selected models
        """
        if len(client_models) <= self.num_malicious + 1:
            # not enough models to apply Krum effectively
            return {
                'selected_models': client_models,
                'selected_indices': list(range(len(client_models))),
                'scores': [0.0] * len(client_models),
                'defense_applied': False
            }
        
        # calc pairwise distances between all models
        distances = self._calculate_pairwise_distances(client_models)
        
        # calc Krum scores for each model
        scores = self._calculate_krum_scores(distances, len(client_models))

        # select models based on Krum criterion
        if self.multi_krum:
            selected_results = self._multi_krum_selection(client_models, scores)
        else:
            selected_results = self._single_krum_selection(client_models, scores)
        
        return selected_results
    
    def _calculate_pairwise_distances(self, client_models):
        """Calculate Euclidean distances between all pairs of models."""
        n_models = len(client_models)
        distances = torch.zeros(n_models, n_models)
        
        # flatten all models for distance calculation
        flattened_models = []
        for model in client_models:
            flattened_models.append(self._flatten_model(model))
        
        # calc pairwise Euclidean distances
        for i in range(n_models):
            for j in range(i + 1, n_models):
                dist = torch.norm(flattened_models[i] - flattened_models[j], p=2)
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _calculate_krum_scores(self, distances, n_models):
        """Calculate Krum scores for each model."""
        scores = []
        n_select = n_models - self.num_malicious - 1
        
        for i in range(n_models):
            # get distances from model i to all other models
            model_distances = distances[i]
            
            # sort distances (excluding self-distance which is 0)
            sorted_distances, _ = torch.sort(model_distances)
            
            # sum the n_select smallest distances (excluding self)
            if n_select >= len(sorted_distances) - 1:
                # if we need to consider all other models
                krum_score = torch.sum(sorted_distances[1:]).item()
            else:
                # sum the n_select smallest distances (excluding self)
                krum_score = torch.sum(sorted_distances[1:n_select + 1]).item()
            
            scores.append(krum_score)
        
        return scores
    
    def _single_krum_selection(self, client_models, scores):
        """Select single model with minimum Krum score."""
        # Find model with minimum score
        min_score_idx = scores.index(min(scores))
        selected_model = client_models[min_score_idx]
        
        return {
            'selected_models': [selected_model],
            'selected_indices': [min_score_idx],
            'scores': scores,
            'defense_applied': True,
            'selection_type': 'single_krum'
        }
    
    def _multi_krum_selection(self, client_models, scores):
        """Select multiple models with lowest Krum scores."""
        # we sort the models by their Krum scores
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        
        # and select top models aka those with lowest scores
        num_select = min(self.num_selected, len(client_models) - self.num_malicious)
        selected_indices = sorted_indices[:num_select]
        selected_models = [client_models[i] for i in selected_indices]
        
        return {
            'selected_models': selected_models,
            'selected_indices': selected_indices,
            'scores': scores,
            'defense_applied': True,
            'selection_type': 'multi_krum'
        }


class TrimmedMean(SimilarityBasedDefense):
    """
    Trimmed Mean Defense: Remove extreme models and average the rest.
    Alternative to Krum that provides averaged result.
    """
    
    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.trim_ratio = defense_params.get('trim_ratio', 0.2)
        self.min_models = defense_params.get('min_models', 3)
    
    def defend(self, client_models, **kwargs):
        """Apply trimmed mean defense."""
        if len(client_models) < self.min_models:
            return {
                'filtered_models': client_models,
                'weights': [1.0 / len(client_models)] * len(client_models),
                'defense_applied': False
            }
        
        # calc center model (median-like)
        center_model = self._find_center_model(client_models)
        
        # calc distances to center
        distances = []
        for i, model in enumerate(client_models):
            dist = torch.norm(self._flatten_model(model) - self._flatten_model(center_model), p=2)
            distances.append((dist.item(), i))
        
        # we sort by distance and trim extremes
        distances.sort()
        n_trim = int(len(client_models) * self.trim_ratio)
        
        # keep middle models
        start_idx = n_trim
        end_idx = len(client_models) - n_trim
        selected_indices = [distances[i][1] for i in range(start_idx, end_idx)]
        selected_models = [client_models[i] for i in selected_indices]
        
        if not selected_models:
            selected_models = [client_models[0]]  # Fallback
            selected_indices = [0]
        
        weights = [1.0 / len(selected_models)] * len(selected_models)
        
        return {
            'filtered_models': selected_models,
            'selected_indices': selected_indices,
            'weights': weights,
            'defense_applied': True,
            'trimmed_count': len(client_models) - len(selected_models)
        }
    
    def _find_center_model(self, client_models):
        """Find model closest to geometric median."""
        n_models = len(client_models)
        if n_models == 1:
            return client_models[0]
        
        # calc all pairwise distances
        total_distances = []
        for i, model_i in enumerate(client_models):
            flat_i = self._flatten_model(model_i)
            total_dist = 0
            
            for j, model_j in enumerate(client_models):
                if i != j:
                    flat_j = self._flatten_model(model_j)
                    dist = torch.norm(flat_i - flat_j, p=2)
                    total_dist += dist.item()
            
            total_distances.append(total_dist)
        
        # reuturn model with minimum total distance (closest to geometric median)
        center_idx = total_distances.index(min(total_distances))
        return client_models[center_idx]


class AdaptiveKrum(Krum):
    """Enhanced Krum with adaptive malicious client estimation."""
    
    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.auto_estimate = defense_params.get('auto_estimate', False)
        self.max_malicious_ratio = defense_params.get('max_malicious_ratio', 0.4)
    
    def defend(self, client_models, **kwargs):
        """Apply adaptive Krum with automatic malicious client estimation."""
        if self.auto_estimate:
            self.num_malicious = self._estimate_malicious_count(client_models)
        
        return super().defend(client_models, **kwargs)
    
    def _estimate_malicious_count(self, client_models):
        """Estimate number of malicious clients based on distance distribution."""
        if len(client_models) < 3:
            return 0
        
        # calc all pairwise distances
        distances = self._calculate_pairwise_distances(client_models)
        
        # fpr each model, calculate its average distance to all others
        avg_distances = []
        for i in range(len(client_models)):
            model_distances = distances[i]
            # exclude self-distance (which is 0)
            non_zero_distances = model_distances[model_distances > 0]
            if len(non_zero_distances) > 0:
                avg_dist = torch.mean(non_zero_distances).item()
            else:
                avg_dist = 0.0
            avg_distances.append(avg_dist)
        
        # we use statistical outlier detection to estimate malicious count
        if len(avg_distances) < 3:
            return 0
        
        mean_dist = sum(avg_distances) / len(avg_distances)
        std_dist = (sum([(d - mean_dist) ** 2 for d in avg_distances]) / len(avg_distances)) ** 0.5
        
        # count models that are statistical outliers (> 2 standard deviations)
        outlier_count = sum([1 for d in avg_distances if abs(d - mean_dist) > 2 * std_dist])
        
        # cap at maximum ratio
        max_malicious = int(len(client_models) * self.max_malicious_ratio)
        estimated_malicious = min(outlier_count, max_malicious)
        
        return estimated_malicious