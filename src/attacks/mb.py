import torch
import copy
import numpy as np
from .base_attack import ModelPoisoningAttack


class ManipulateByzantineAttack(ModelPoisoningAttack):
    """
    Manipulate Byzantine (MB) Attack: 
    Iterative method to find shared scalar for poisoning all parameters.
    """
    
    def __init__(self, attack_params=None):
        super().__init__(attack_params)
        self.max_iterations = attack_params.get('max_iterations', 100)
        self.step_size = attack_params.get('step_size', 0.1)
        self.target_objective = attack_params.get('target_objective', 'maximize_distance')
        self.scaling_bounds = attack_params.get('scaling_bounds', (-50.0, 50.0))
    
    def attack(self, model, benign_models=None, **kwargs):
        """
        MB Attack: Find shared scalar to manipulate all parameters.
        
        Args:
            model: The local model to be poisoned
            benign_models: List of benign models for reference (if available)
        
        Returns:
            dict: Attack results including metrics
        """
        original_model = copy.deepcopy(model)
        
        # find optimal shared scalar
        optimal_scalar = self._find_shared_scalar(model, benign_models)
        
        # apply attack with optimal scalar
        self._apply_scaling_attack(model, optimal_scalar)
        
        # calc attack metrics
        metrics = self.calculate_similarity_metrics(original_model, model)
        
        return {
            'attack_success': True,
            'metrics': metrics,
            'scaling_factor': optimal_scalar
        }
    
    def _find_shared_scalar(self, model, benign_models=None):
        """Find optimal shared scalar using iterative search."""
        best_scalar = -1.0  # Start with simple negation
        best_objective = float('-inf')
        
        # def search space
        min_scalar, max_scalar = self.scaling_bounds
        scalars = np.linspace(min_scalar, max_scalar, self.max_iterations)
        
        for scalar in scalars:
            # we re skipping zero scaling (no attack)
            if abs(scalar) < 1e-6:
                continue
            
            # temporary poisoned model
            temp_model = copy.deepcopy(model)
            self._apply_scaling_attack(temp_model, scalar)
            
            # calc objective based on target
            if self.target_objective == 'maximize_distance':
                objective = self._calculate_distance_objective(model, temp_model)
            elif self.target_objective == 'minimize_similarity':
                objective = self._calculate_similarity_objective(model, temp_model)
            elif self.target_objective == 'maximize_norm_ratio':
                objective = self._calculate_norm_ratio_objective(model, temp_model)
            else:
                objective = scalar
            
            if objective > best_objective:
                best_objective = objective
                best_scalar = scalar
        
        return best_scalar
    
    def _apply_scaling_attack(self, model, scalar):
        """Apply uniform scaling to all model parameters."""
        for param in model.parameters():
            param.data = param.data * scalar
    
    def _calculate_distance_objective(self, original_model, poisoned_model):
        """Calculate Euclidean distance between models."""
        flat_orig = self._flatten_model(original_model)
        flat_poison = self._flatten_model(poisoned_model)
        distance = torch.norm(flat_orig - flat_poison, p=2)
        return distance.item()
    
    def _calculate_similarity_objective(self, original_model, poisoned_model):
        """Calculate negative cosine similarity (to maximize dissimilarity)."""
        flat_orig = self._flatten_model(original_model)
        flat_poison = self._flatten_model(poisoned_model)
        
        dot_product = torch.dot(flat_orig, flat_poison)
        norm_orig = torch.norm(flat_orig, p=2)
        norm_poison = torch.norm(flat_poison, p=2)
        
        if norm_orig == 0 or norm_poison == 0:
            return 1.0  # max dissimilarity
        
        cosine_sim = dot_product / (norm_orig * norm_poison)
        return -cosine_sim.item()  # negative for maximization
    
    def _calculate_norm_ratio_objective(self, original_model, poisoned_model):
        """Calculate norm ratio between poisoned and original models."""
        norm_orig = self._get_model_norm(original_model)
        norm_poison = self._get_model_norm(poisoned_model)
        
        if norm_orig == 0:
            return float('inf')
        
        return norm_poison / norm_orig
    
    def _get_model_norm(self, model, p=2):
        """Calculate model norm."""
        total_norm = 0
        for param in model.parameters():
            param_norm = param.data.norm(p)
            total_norm += param_norm.item() ** p
        return total_norm ** (1.0 / p)


class AdaptiveMBAttack(ManipulateByzantineAttack):
    """Enhanced MB Attack with adaptive search and multiple objectives."""
    
    def __init__(self, attack_params=None):
        super().__init__(attack_params)
        self.multi_objective = attack_params.get('multi_objective', False)
        self.objective_weights = attack_params.get('objective_weights', [1.0, 1.0, 1.0])
    
    def _find_shared_scalar(self, model, benign_models=None):
        """Enhanced scalar search with adaptive refinement."""
        # Coarse search
        coarse_scalars = np.logspace(-2, 2, 50)  # from 0.01 to 100
        coarse_scalars = np.concatenate([-coarse_scalars, coarse_scalars])
        
        best_scalar = -1.0
        best_objective = float('-inf')
        
        # Coarse search
        for scalar in coarse_scalars:
            if abs(scalar) < 1e-6:
                continue
                
            temp_model = copy.deepcopy(model)
            self._apply_scaling_attack(temp_model, scalar)
            
            if self.multi_objective:
                objective = self._calculate_multi_objective(model, temp_model)
            else:
                objective = self._calculate_distance_objective(model, temp_model)
            
            if objective > best_objective:
                best_objective = objective
                best_scalar = scalar
        
        fine_range = abs(best_scalar) * 0.2
        fine_scalars = np.linspace(best_scalar - fine_range, best_scalar + fine_range, 20)
        
        for scalar in fine_scalars:
            if abs(scalar) < 1e-6:
                continue
                
            temp_model = copy.deepcopy(model)
            self._apply_scaling_attack(temp_model, scalar)
            
            if self.multi_objective:
                objective = self._calculate_multi_objective(model, temp_model)
            else:
                objective = self._calculate_distance_objective(model, temp_model)
            
            if objective > best_objective:
                best_objective = objective
                best_scalar = scalar
        
        return best_scalar
    
    def _calculate_multi_objective(self, original_model, poisoned_model):
        """Calculate weighted combination of multiple objectives."""
        distance_obj = self._calculate_distance_objective(original_model, poisoned_model)
        similarity_obj = self._calculate_similarity_objective(original_model, poisoned_model)
        norm_ratio_obj = self._calculate_norm_ratio_objective(original_model, poisoned_model)
        
        # normalize objectives
        distance_obj = min(distance_obj / 1000.0, 1.0)  # Normalize distance
        similarity_obj = min(abs(similarity_obj), 1.0)   # Normalize similarity
        norm_ratio_obj = min(norm_ratio_obj / 10.0, 1.0) # Normalize norm ratio
        
        # weighted combination
        total_objective = (self.objective_weights[0] * distance_obj +
                          self.objective_weights[1] * similarity_obj +
                          self.objective_weights[2] * norm_ratio_obj)
        
        return total_objective