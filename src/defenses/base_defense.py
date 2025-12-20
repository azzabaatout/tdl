import torch
import copy
from abc import ABC, abstractmethod


class BaseDefense(ABC):
    def __init__(self, defense_params=None):
        self.defense_params = defense_params or {}
        self.defense_name = self.__class__.__name__
    
    @abstractmethod
    def defend(self, client_models, **kwargs):
        """Apply defense mechanism to client models."""
        pass
    
    def get_defense_name(self):
        """Get name of the defense."""
        return self.defense_name


class SimilarityBasedDefense(BaseDefense):
    """Base class for similarity-based defenses."""
    
    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.similarity_threshold = defense_params.get('similarity_threshold', 0.5)
        self.use_cosine = defense_params.get('use_cosine', True)
        self.use_euclidean = defense_params.get('use_euclidean', True)
        self.use_l2_norm = defense_params.get('use_l2_norm', True)
    
    def calculate_similarity_metrics(self, model1, model2):
        """Calculate similarity metrics between two models."""
        flat1 = self._flatten_model(model1)
        flat2 = self._flatten_model(model2)
        
        metrics = {}
        
        if self.use_l2_norm:
            # L2 norm ratio
            l2_norm1 = torch.norm(flat1, p=2)
            l2_norm2 = torch.norm(flat2, p=2)
            metrics['l2_ratio'] = (l2_norm1 / l2_norm2).item() if l2_norm2 != 0 else float('inf')
        
        if self.use_euclidean:
            euclidean_dist = torch.norm(flat1 - flat2, p=2)
            metrics['euclidean_distance'] = euclidean_dist.item()
        
        if self.use_cosine:
            dot_product = torch.dot(flat1, flat2)
            norm1 = torch.norm(flat1, p=2)
            norm2 = torch.norm(flat2, p=2)
            
            if norm1 != 0 and norm2 != 0:
                cosine_sim = (dot_product / (norm1 * norm2)).item()
            else:
                cosine_sim = 0.0
            metrics['cosine_similarity'] = cosine_sim
        
        return metrics
    
    def _flatten_model(self, model):
        """Flatten model parameters into a 1D tensor."""
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)
    
    def _unflatten_model(self, flattened_params, model):
        """Unflatten 1D tensor back into model parameters."""
        pointer = 0
        for param in model.parameters():
            num_params = param.numel()
            param.data = flattened_params[pointer:pointer + num_params].view(param.shape)
            pointer += num_params
    
    def is_model_malicious(self, model, reference_model, threshold_multiplier=1.0):
        """Determine if a model is malicious based on similarity metrics."""
        if reference_model is None:
            return False
        
        metrics = self.calculate_similarity_metrics(model, reference_model)
        
        # we re checking various similarity criteria
        is_malicious = False
        
        if self.use_cosine and 'cosine_similarity' in metrics:
            if metrics['cosine_similarity'] < self.similarity_threshold * threshold_multiplier:
                is_malicious = True
        
        if self.use_l2_norm and 'l2_ratio' in metrics:
            # we re considering malicious if norm ratio is too high or too low
            l2_ratio = metrics['l2_ratio']
            if l2_ratio > 5.0 or l2_ratio < 0.2:
                is_malicious = True
        
        if self.use_euclidean and 'euclidean_distance' in metrics:
            # this would probably need domain-specific thresholding
            pass
        
        return is_malicious