import torch
import copy
from .base_defense import SimilarityBasedDefense


class NormClipping(SimilarityBasedDefense):
    """
    Norm Clipping Defense: Reject models with L2 norm exceeding threshold.
    Simple but effective against magnitude-based attacks.
    """
    
    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.max_norm = defense_params.get('max_norm', 10.0)
        self.adaptive_threshold = defense_params.get('adaptive_threshold', False)
        self.norm_type = defense_params.get('norm_type', 2)  # L1 or L2 norm
        self.relative_threshold = defense_params.get('relative_threshold', False)
    
    def defend(self, client_models, global_model=None, **kwargs):
        """
        Apply norm clipping defense.
        
        Args:
            client_models: List of client models
            global_model: Reference global model for relative thresholding
        
        Returns:
            dict: Defense results with filtered models
        """
        if self.adaptive_threshold:
            threshold = self._calculate_adaptive_threshold(client_models, global_model)
        elif self.relative_threshold and global_model is not None:
            threshold = self._calculate_relative_threshold(global_model)
        else:
            threshold = self.max_norm
        
        # filter models based on norm threshold
        filtered_models = []
        filtered_indices = []
        model_norms = []
        rejected_count = 0
        
        for i, model in enumerate(client_models):
            model_norm = self._calculate_model_norm(model)
            model_norms.append(model_norm)
            
            if model_norm <= threshold:
                filtered_models.append(model)
                filtered_indices.append(i)
            else:
                rejected_count += 1
        
        # if all the models are rejected, we keep the one with smallest norm
        if not filtered_models:
            min_norm_idx = model_norms.index(min(model_norms))
            filtered_models = [client_models[min_norm_idx]]
            filtered_indices = [min_norm_idx]
            rejected_count = len(client_models) - 1
        
        # equal weights for all accepted models
        weights = [1.0 / len(filtered_models)] * len(filtered_models)
        
        return {
            'filtered_models': filtered_models,
            'selected_indices': filtered_indices,
            'weights': weights,
            'rejected_count': rejected_count,
            'threshold_used': threshold,
            'model_norms': model_norms,
            'defense_applied': True
        }
    
    def _calculate_model_norm(self, model):
        """Calculate the L1 or L2 norm of model parameters."""
        total_norm = 0.0
        
        for param in model.parameters():
            if self.norm_type == 1:
                param_norm = torch.sum(torch.abs(param.data))
            else:
                param_norm = torch.norm(param.data, p=2) ** 2
            total_norm += param_norm.item()
        
        if self.norm_type == 2:
            total_norm = total_norm ** 0.5
        
        return total_norm
    
    def _calculate_adaptive_threshold(self, client_models, global_model=None):
        """Calculate adaptive threshold based on model norm distribution."""
        model_norms = [self._calculate_model_norm(model) for model in client_models]
        
        if len(model_norms) < 2:
            return self.max_norm
        
        # Calculate statistics
        mean_norm = sum(model_norms) / len(model_norms)
        std_norm = (sum([(norm - mean_norm) ** 2 for norm in model_norms]) / len(model_norms)) ** 0.5
        
        # set threshold as mean + 2*std and allows some variation but rejects outliers
        adaptive_threshold = mean_norm + 2 * std_norm
        
        # we make sure that we don t let adaptive threshold be too permissive
        return min(adaptive_threshold, self.max_norm * 2)
    
    def _calculate_relative_threshold(self, global_model):
        """Calculate threshold relative to global model norm."""
        global_norm = self._calculate_model_norm(global_model)
        # allow models to be up to 5x the global model norm
        relative_threshold = global_norm * 5.0
        return min(relative_threshold, self.max_norm)


class GradientClipping(SimilarityBasedDefense):
    """
    Gradient Clipping Defense: Clip model updates rather than absolute norms.
    More sophisticated than norm clipping as it considers the update magnitude.
    """
    
    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.max_update_norm = defense_params.get('max_update_norm', 5.0)
        self.clip_type = defense_params.get('clip_type', 'norm')
        self.adaptive_clipping = defense_params.get('adaptive_clipping', True)
    
    def defend(self, client_models, global_model=None, **kwargs):
        """
        Apply gradient clipping defense.
        
        Args:
            client_models: List of client models
            global_model: Previous global model to compute updates
        
        Returns:
            dict: Defense results with clipped models
        """
        if global_model is None:
            # if no global model, fall back to norm clipping
            norm_clipper = NormClipping({'max_norm': self.max_update_norm})
            return norm_clipper.defend(client_models, **kwargs)
        
        clipped_models = []
        update_norms = []
        clipped_count = 0
        
        # calc adaptive threshold if enabled
        if self.adaptive_clipping:
            threshold = self._calculate_adaptive_update_threshold(client_models, global_model)
        else:
            threshold = self.max_update_norm
        
        for model in client_models:
            clipped_model, update_norm, was_clipped = self._clip_model_update(
                model, global_model, threshold
            )
            clipped_models.append(clipped_model)
            update_norms.append(update_norm)
            if was_clipped:
                clipped_count += 1
        
        weights = [1.0 / len(clipped_models)] * len(clipped_models)
        
        return {
            'filtered_models': clipped_models,
            'weights': weights,
            'clipped_count': clipped_count,
            'update_norms': update_norms,
            'threshold_used': threshold,
            'defense_applied': True
        }
    
    def _clip_model_update(self, client_model, global_model, threshold):
        """Clip the update from global model to client model."""
        clipped_model = copy.deepcopy(client_model)
        
        # calc update vector
        update_norm = 0.0
        updates = []
        
        for client_param, global_param in zip(client_model.parameters(), global_model.parameters()):
            update = client_param.data - global_param.data
            updates.append(update)
            update_norm += torch.norm(update, p=2).item() ** 2
        
        update_norm = update_norm ** 0.5
        
        if update_norm <= threshold:
            return clipped_model, update_norm, False
        
        # apply clipping
        scaling_factor = threshold / update_norm
        
        for clipped_param, global_param, update in zip(clipped_model.parameters(), 
                                                     global_model.parameters(), 
                                                     updates):
            clipped_update = update * scaling_factor
            clipped_param.data = global_param.data + clipped_update
        
        return clipped_model, update_norm, True
    
    def _calculate_adaptive_update_threshold(self, client_models, global_model):
        """Calculate adaptive threshold based on update norm distribution."""
        update_norms = []
        
        for model in client_models:
            update_norm = 0.0
            for client_param, global_param in zip(model.parameters(), global_model.parameters()):
                update = client_param.data - global_param.data
                update_norm += torch.norm(update, p=2).item() ** 2
            update_norms.append(update_norm ** 0.5)
        
        if len(update_norms) < 2:
            return self.max_update_norm
        
        # use median + std as threshold
        update_norms.sort()
        median_norm = update_norms[len(update_norms) // 2]
        mean_norm = sum(update_norms) / len(update_norms)
        std_norm = (sum([(norm - mean_norm) ** 2 for norm in update_norms]) / len(update_norms)) ** 0.5
        
        adaptive_threshold = median_norm + 2 * std_norm
        return min(adaptive_threshold, self.max_update_norm * 3)


class HybridNormDefense(SimilarityBasedDefense):
    """
    Hybrid defense combining norm clipping with additional filters.
    Uses multiple criteria to detect and filter malicious models.
    """
    
    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.norm_threshold = defense_params.get('norm_threshold', 10.0)
        self.cosine_threshold = defense_params.get('cosine_threshold', -0.5)
        self.use_statistical_filter = defense_params.get('use_statistical_filter', True)
        self.std_multiplier = defense_params.get('std_multiplier', 2.0)
    
    def defend(self, client_models, global_model=None, **kwargs):
        """Apply hybrid norm-based defense with multiple filters."""
        # step 1: Basic norm clipping
        norm_results = self._apply_norm_filter(client_models)
        
        # step 2: Cosine similarity filter (if global model available)
        if global_model is not None:
            cosine_results = self._apply_cosine_filter(
                norm_results['filtered_models'], global_model
            )
        else:
            cosine_results = norm_results
        
        # step 3: Statistical outlier filter
        if self.use_statistical_filter and len(cosine_results['filtered_models']) > 2:
            final_results = self._apply_statistical_filter(cosine_results['filtered_models'])
        else:
            final_results = cosine_results
        
        # combine results
        total_rejected = (len(client_models) - len(norm_results['filtered_models']) +
                         len(norm_results['filtered_models']) - len(cosine_results['filtered_models']) +
                         len(cosine_results['filtered_models']) - len(final_results['filtered_models']))
        
        weights = [1.0 / len(final_results['filtered_models'])] * len(final_results['filtered_models'])
        
        return {
            'filtered_models': final_results['filtered_models'],
            'weights': weights,
            'rejected_count': total_rejected,
            'defense_applied': True,
            'filter_stages': {
                'norm_filter': len(client_models) - len(norm_results['filtered_models']),
                'cosine_filter': len(norm_results['filtered_models']) - len(cosine_results['filtered_models']),
                'statistical_filter': len(cosine_results['filtered_models']) - len(final_results['filtered_models'])
            }
        }
    
    def _apply_norm_filter(self, client_models):
        """Apply basic norm filtering."""
        norm_clipper = NormClipping({'max_norm': self.norm_threshold})
        return norm_clipper.defend(client_models)
    
    def _apply_cosine_filter(self, client_models, global_model):
        """Filter based on cosine similarity to global model."""
        global_flat = self._flatten_model(global_model)
        filtered_models = []
        
        for model in client_models:
            model_flat = self._flatten_model(model)
            cosine_sim = self._calculate_cosine_similarity(model_flat, global_flat)
            
            if cosine_sim >= self.cosine_threshold:
                filtered_models.append(model)
        
        if not filtered_models:
            filtered_models = client_models
        
        return {'filtered_models': filtered_models}
    
    def _apply_statistical_filter(self, client_models):
        """Filter statistical outliers based on pairwise distances."""
        if len(client_models) <= 2:
            return {'filtered_models': client_models}
        
        # calc average distance from each model to all others
        avg_distances = []
        for i, model_i in enumerate(client_models):
            flat_i = self._flatten_model(model_i)
            total_dist = 0
            
            for j, model_j in enumerate(client_models):
                if i != j:
                    flat_j = self._flatten_model(model_j)
                    dist = torch.norm(flat_i - flat_j, p=2).item()
                    total_dist += dist
            
            avg_dist = total_dist / (len(client_models) - 1)
            avg_distances.append(avg_dist)
        
        # filter outliers
        mean_dist = sum(avg_distances) / len(avg_distances)
        std_dist = (sum([(d - mean_dist) ** 2 for d in avg_distances]) / len(avg_distances)) ** 0.5
        
        filtered_models = []
        for i, avg_dist in enumerate(avg_distances):
            z_score = abs(avg_dist - mean_dist) / std_dist if std_dist > 0 else 0
            if z_score <= self.std_multiplier:
                filtered_models.append(client_models[i])
        
        if not filtered_models:
            filtered_models = client_models
        
        return {'filtered_models': filtered_models}
    
    def _calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = torch.dot(vec1, vec2)
        norm1 = torch.norm(vec1, p=2)
        norm2 = torch.norm(vec2, p=2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()