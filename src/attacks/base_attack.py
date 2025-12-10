import torch
import copy
from abc import ABC, abstractmethod


class BaseAttack(ABC):
    def __init__(self, attack_params=None):
        self.attack_params = attack_params or {}
    
    @abstractmethod
    def attack(self, model, **kwargs):
        pass
    
    def get_attack_name(self):
        return self.__class__.__name__


class ModelPoisoningAttack(BaseAttack):
    def __init__(self, attack_params=None):
        super().__init__(attack_params)
        self.attack_type = "model_poisoning"
    
    def calculate_similarity_metrics(self, model1, model2):
        flat1 = self._flatten_model(model1)
        flat2 = self._flatten_model(model2)
        
        # L2 norm ratio
        l2_norm1 = torch.norm(flat1, p=2)
        l2_norm2 = torch.norm(flat2, p=2)
        l2_ratio = l2_norm1 / l2_norm2 if l2_norm2 != 0 else float('inf')
        
        # euc distance
        euclidean_dist = torch.norm(flat1 - flat2, p=2)
        
        # cos sim
        dot_product = torch.dot(flat1, flat2)
        cosine_sim = dot_product / (l2_norm1 * l2_norm2) if l2_norm1 != 0 and l2_norm2 != 0 else 0
        
        return {
            'l2_ratio': l2_ratio.item(),
            'euclidean_distance': euclidean_dist.item(),
            'cosine_similarity': cosine_sim.item()
        }
    
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