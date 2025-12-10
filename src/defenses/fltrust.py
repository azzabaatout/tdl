import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from .base_defense import SimilarityBasedDefense


class FLTrust(SimilarityBasedDefense):
    """
    FLTrust Defense: Server maintains clean dataset and uses it to evaluate client updates.
    Based on cosine similarity and L2 norm analysis.
    """
    
    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.server_epochs = defense_params.get('server_epochs', 5)
        self.server_lr = defense_params.get('server_lr', 0.01)
        self.cosine_threshold = defense_params.get('cosine_threshold', 0.0)
        self.use_relu = defense_params.get('use_relu', True)
        self.clip_updates = defense_params.get('clip_updates', True)
        
    def defend(self, client_models, server_data=None, global_model=None, **kwargs):
        """
        Apply FLTrust defense mechanism.
        
        Args:
            client_models: List of client models
            server_data: Clean dataset on server
            global_model: Current global model
        
        Returns:
            dict: Defense results with filtered models and weights
        """
        if server_data is None or global_model is None:
            # If no server data available, fall back to simple averaging
            return {
                'filtered_models': client_models,
                'weights': [1.0 / len(client_models)] * len(client_models),
                'rejected_count': 0,
                'defense_applied': False
            }
        
        # Step 1 - train server model on clean data
        server_model = self._train_server_model(global_model, server_data)
        
        # Step 2 - eval client models using server model
        model_scores = self._evaluate_client_models(client_models, server_model)
        
        # Step 3 - filter and weight client models
        filtered_results = self._filter_and_weight_models(client_models, model_scores)
        
        return filtered_results
    
    def _train_server_model(self, global_model, server_data):
        """Train a clean model on server data."""
        server_model = copy.deepcopy(global_model)
        device = next(server_model.parameters()).device
        
        server_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(server_model.parameters(), lr=self.server_lr, momentum=0.9)
        
        server_loader = DataLoader(server_data, batch_size=32, shuffle=True)
        
        for epoch in range(self.server_epochs):
            for batch_idx, (data, target) in enumerate(server_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = server_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return server_model
    
    def _evaluate_client_models(self, client_models, server_model):
        """Evaluate client models against server model using cosine similarity and L2 norm."""
        server_flat = self._flatten_model(server_model)
        model_scores = []
        
        for i, client_model in enumerate(client_models):
            client_flat = self._flatten_model(client_model)
            
            cosine_sim = self._calculate_cosine_similarity(client_flat, server_flat)
            
            # aplly relu to filter out opposite directions
            if self.use_relu:
                cosine_score = max(0.0, cosine_sim)
            else:
                cosine_score = cosine_sim
            
            # calc L2 norm ratio for magnitude analysis
            client_norm = torch.norm(client_flat, p=2)
            server_norm = torch.norm(server_flat, p=2)
            
            if server_norm != 0:
                norm_ratio = client_norm / server_norm
            else:
                norm_ratio = float('inf')
            
            # fltrust uses cosine similarity as primary filter
            final_score = cosine_score
            
            model_scores.append({
                'model_idx': i,
                'cosine_similarity': cosine_sim,
                'cosine_score': cosine_score,
                'norm_ratio': norm_ratio.item(),
                'final_score': final_score
            })
        
        return model_scores
    
    def _calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = torch.dot(vec1, vec2)
        norm1 = torch.norm(vec1, p=2)
        norm2 = torch.norm(vec2, p=2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()
    
    def _filter_and_weight_models(self, client_models, model_scores):
        """Filter malicious models and compute weights for remaining models."""
        # filter out models based on cosine threshold
        valid_models = []
        valid_scores = []
        rejected_count = 0
        
        for score_info in model_scores:
            if score_info['cosine_score'] >= self.cosine_threshold:
                valid_models.append(client_models[score_info['model_idx']])
                valid_scores.append(score_info)
            else:
                rejected_count += 1
        
        if not valid_models:
            # if all models rejected, use all with equal weights
            return {
                'filtered_models': client_models,
                'weights': [1.0 / len(client_models)] * len(client_models),
                'rejected_count': 0,
                'defense_applied': True,
                'all_rejected': True
            }
        
        # compute weights based on cosine scores
        scores = [score_info['final_score'] for score_info in valid_scores]
        total_score = sum(scores)
        
        if total_score == 0:
            weights = [1.0 / len(valid_models)] * len(valid_models)
        else:
            weights = [score / total_score for score in scores]
        
        # Optional but might be helpful: we can apply norm clipping to weights

        if self.clip_updates:
            weights = self._clip_weights(weights, valid_scores)
        
        return {
            'filtered_models': valid_models,
            'weights': weights,
            'rejected_count': rejected_count,
            'defense_applied': True,
            'model_scores': valid_scores
        }
    
    def _clip_weights(self, weights, score_infos):
        """Clip weights based on norm ratios to reduce impact of magnitude attacks."""
        clipped_weights = []
        
        for i, (weight, score_info) in enumerate(zip(weights, score_infos)):
            norm_ratio = score_info['norm_ratio']
            
            # Clip weights if norm ratio is too large
            if norm_ratio > 5.0:  # Threshold for large norms
                clipped_weight = weight * min(1.0, 5.0 / norm_ratio)
            else:
                clipped_weight = weight
            
            clipped_weights.append(clipped_weight)
        
        # Renormalize weights
        total_weight = sum(clipped_weights)
        if total_weight > 0:
            clipped_weights = [w / total_weight for w in clipped_weights]
        else:
            clipped_weights = [1.0 / len(clipped_weights)] * len(clipped_weights)
        
        return clipped_weights


class EnhancedFLTrust(FLTrust):
    """Enhanced FLTrust with additional filtering mechanisms."""
    
    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.use_statistical_filtering = defense_params.get('use_statistical_filtering', True)
        self.std_threshold = defense_params.get('std_threshold', 2.0)
    
    def _filter_and_weight_models(self, client_models, model_scores):
        """Enhanced filtering with statistical outlier detection."""
        # First apply standard FLTrust filtering
        initial_results = super()._filter_and_weight_models(client_models, model_scores)
        
        if not self.use_statistical_filtering or len(initial_results['filtered_models']) <= 2:
            return initial_results
        
        # additional statistical filtering
        valid_scores = initial_results['model_scores']
        cosine_scores = [score['cosine_score'] for score in valid_scores]
        norm_ratios = [score['norm_ratio'] for score in valid_scores]
        
        cosine_mean = sum(cosine_scores) / len(cosine_scores)
        cosine_std = (sum([(x - cosine_mean) ** 2 for x in cosine_scores]) / len(cosine_scores)) ** 0.5
        
        norm_mean = sum(norm_ratios) / len(norm_ratios)
        norm_std = (sum([(x - norm_mean) ** 2 for x in norm_ratios]) / len(norm_ratios)) ** 0.5
        
        # filter statistical outliers
        final_models = []
        final_scores = []
        additional_rejected = 0
        
        for i, score_info in enumerate(valid_scores):
            cosine_z_score = abs(score_info['cosine_score'] - cosine_mean) / cosine_std if cosine_std > 0 else 0
            norm_z_score = abs(score_info['norm_ratio'] - norm_mean) / norm_std if norm_std > 0 else 0
            
            # keep if not an outlier in either metric
            if cosine_z_score <= self.std_threshold and norm_z_score <= self.std_threshold:
                final_models.append(initial_results['filtered_models'][i])
                final_scores.append(score_info)
            else:
                additional_rejected += 1
        
        if not final_models:
            return initial_results  # Fall back to initial results
        
        # recompute weights
        scores = [score_info['final_score'] for score_info in final_scores]
        total_score = sum(scores)
        
        if total_score == 0:
            final_weights = [1.0 / len(final_models)] * len(final_models)
        else:
            final_weights = [score / total_score for score in scores]
        
        return {
            'filtered_models': final_models,
            'weights': final_weights,
            'rejected_count': initial_results['rejected_count'] + additional_rejected,
            'defense_applied': True,
            'model_scores': final_scores,
            'statistical_filtering': True
        }