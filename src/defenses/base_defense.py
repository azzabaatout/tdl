import torch
from abc import ABC, abstractmethod

class BaseDefense(ABC):
    def __init__(self, defense_params=None):
        self.defense_params = defense_params or {}
        self.defense_name = self.__class__.__name__

    @abstractmethod
    def defend(self, client_models, **kwargs):
        """
        Apply defense mechanism.
        Must return a dict containing 'filtered_models' and 'weights'.
        """
        pass

class SimilarityBasedDefense(BaseDefense):
    """
    Base class providing similarity metric primitives used in paper defenses.
    Does NOT enforce arbitrary thresholds (that is left to subclasses).
    """

    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        # no hardcoded thresholds here - let subclasses decide

    def _flatten_model(self, model):
        """Flatten model parameters into a single 1D tensor."""
        # crual for paper's vector-based similarity definitions
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def compute_cosine_similarity(self, v1, v2):
        """Paper definition: (v1 . v2) / (||v1|| * ||v2||)"""
        norm1 = torch.norm(v1, p=2)
        norm2 = torch.norm(v2, p=2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return torch.dot(v1, v2) / (norm1 * norm2)

    def compute_euclidean_distance(self, v1, v2):
        """Paper definition: ||v1 - v2||"""
        return torch.norm(v1 - v2, p=2)

    def compute_l2_norm(self, v):
        """Paper definition: ||v||"""
        return torch.norm(v, p=2)
