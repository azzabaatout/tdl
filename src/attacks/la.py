import torch
import copy
import numpy as np
from .base_attack import ModelPoisoningAttack


class LocalAttack(ModelPoisoningAttack):
    def __init__(self, attack_params=None):
        super().__init__(attack_params)
        self.max_iterations = attack_params.get('max_iterations', 100)
        self.step_size = attack_params.get('step_size', 0.1)
        self.target_norm_ratio = attack_params.get('target_norm_ratio', 10.0)
    
    def attack(self, model, benign_models=None, **kwargs):
        """
        Local Attack (LA): Maximize parameter direction changes while maintaining certain constraints.
        
        Args:
            model: The local model to be poisoned
            benign_models: List of benign models for reference (if available)
        
        Returns:
            dict: Attack results including metrics
        """
        original_model = copy.deepcopy(model)
        
        # step 1: gen random direction vector
        direction_vector = self._generate_direction_vector(model)
        
        # step 2: find optimal scaling factor
        optimal_scalar = self._find_optimal_scalar(model, direction_vector)
        
        # step 3: apply attack
        self._apply_directional_poisoning(model, direction_vector, optimal_scalar)
        
        # Calculate attack metrics
        metrics = self.calculate_similarity_metrics(original_model, model)
        
        return {
            'attack_success': True,
            'metrics': metrics,
            'scaling_factor': optimal_scalar
        }
    
    def _generate_direction_vector(self, model):
        """Generate random direction vector with +1 and -1."""
        direction_dict = {}
        
        for name, param in model.named_parameters():
            # gen random signs (1 or -1) for each parameter
            random_signs = torch.randint(0, 2, param.shape, device=param.device, dtype=torch.float32)
            random_signs = random_signs * 2 - 1  # Convert {0, 1} to {-1, 1}
            direction_dict[name] = random_signs
        
        return direction_dict
    
    def _find_optimal_scalar(self, model, direction_vector):
        """Find optimal scalar using iterative method."""
        best_scalar = 1.0
        best_objective = float('-inf')
        
        # try out different scaling factors
        scalars = np.logspace(-2, 2, self.max_iterations)  # From 0.01 to 100
        
        for scalar in scalars:
            # create temporary model with this scalar
            temp_model = copy.deepcopy(model)
            self._apply_directional_poisoning(temp_model, direction_vector, scalar)
            
            # calc objective (maximize norm ratio)
            original_norm = self._get_model_norm(model)
            poisoned_norm = self._get_model_norm(temp_model)
            norm_ratio = poisoned_norm / original_norm if original_norm != 0 else float('inf')
            
            # objective is: maximize norm ratio while keeping it reasonable
            if norm_ratio > best_objective and norm_ratio <= self.target_norm_ratio:
                best_objective = norm_ratio
                best_scalar = scalar
        
        return best_scalar
    
    def _apply_directional_poisoning(self, model, direction_vector, scalar):
        """Apply directional poisoning with given scalar."""
        for name, param in model.named_parameters():
            if name in direction_vector:
                # param = param + scalar * direction * |param|
                direction = direction_vector[name]
                magnitude = torch.abs(param.data)
                poisoning = scalar * direction * magnitude
                param.data = param.data + poisoning
    
    def _get_model_norm(self, model, p=2):
        """Calculate model norm."""
        total_norm = 0
        for param in model.parameters():
            param_norm = param.data.norm(p)
            total_norm += param_norm.item() ** p
        return total_norm ** (1.0 / p)


class AdaptiveLocalAttack(LocalAttack):
    """Enhanced Local Attack with adaptive scaling."""
    
    def __init__(self, attack_params=None):
        super().__init__(attack_params)
        self.adaptive_steps = attack_params.get('adaptive_steps', 10)
        self.convergence_threshold = attack_params.get('convergence_threshold', 1e-4)
    
    def _find_optimal_scalar(self, model, direction_vector):
        """Find optimal scalar using adaptive binary search."""
        left, right = 0.1, 10.0
        best_scalar = 1.0
        
        for _ in range(self.adaptive_steps):
            mid = (left + right) / 2
            
            # test this scalar
            temp_model = copy.deepcopy(model)
            self._apply_directional_poisoning(temp_model, direction_vector, mid)
            
            # calc metrics
            original_norm = self._get_model_norm(model)
            poisoned_norm = self._get_model_norm(temp_model)
            norm_ratio = poisoned_norm / original_norm if original_norm != 0 else float('inf')
            
            if norm_ratio < self.target_norm_ratio:
                left = mid
                best_scalar = mid
            else:
                right = mid
            
            if abs(right - left) < self.convergence_threshold:
                break
        
        return best_scalar