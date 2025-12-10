import torch
import copy
import numpy as np
from .base_attack import ModelPoisoningAttack


class FakerAttack(ModelPoisoningAttack):
    """
    Faker Attack: Sophisticated attack that exploits vulnerabilities in similarity metrics.
    Based on the paper "Can We Trust the Similarity Measurement in Federated Learning".
    """
    
    def __init__(self, attack_params=None):
        super().__init__(attack_params)
        self.max_iterations = attack_params.get('max_iterations', 100)
        self.learning_rate = attack_params.get('learning_rate', 0.01)
        self.target_similarity = attack_params.get('target_similarity', 0.9)
        self.similarity_type = attack_params.get('similarity_type', 'cosine')
        self.stealth_factor = attack_params.get('stealth_factor', 1.0)
        self.convergence_threshold = attack_params.get('convergence_threshold', 1e-4)
    
    def attack(self, model, benign_models=None, reference_model=None, **kwargs):
        """
        Faker Attack: Generate poisoned model that appears similar to benign models.
        
        Args:
            model: The local model to be poisoned
            benign_models: List of benign models for similarity calculation
            reference_model: Reference model (e.g., previous global model)
        
        Returns:
            dict: Attack results including metrics
        """
        original_model = copy.deepcopy(model)
        
        if reference_model is None and benign_models:
            # use first benign model as reference if no explicit reference
            reference_model = benign_models[0]
        elif reference_model is None:
            # if no reference available, create a simple poisoned model
            return self._simple_poisoning(model, original_model)
        
        # we execute sophisticated similarity-based attack
        attack_success = self._execute_similarity_attack(model, reference_model, benign_models)
        
        # and calculate attack metrics
        metrics = self.calculate_similarity_metrics(original_model, model)
        
        return {
            'attack_success': attack_success,
            'metrics': metrics,
            'similarity_type': self.similarity_type
        }
    
    def _execute_similarity_attack(self, model, reference_model, benign_models=None):
        target_flat = self._flatten_model(reference_model)
        current_flat = self._flatten_model(model)
        
        # init attack direction
        attack_direction = torch.randn_like(current_flat)
        attack_direction = attack_direction / torch.norm(attack_direction)
        
        best_loss = float('inf')
        best_params = current_flat.clone()
        
        for iteration in range(self.max_iterations):
            # cacl gradient for similarity objective
            grad = self._calculate_similarity_gradient(current_flat, target_flat, attack_direction)
            
            # update of params
            current_flat = current_flat - self.learning_rate * grad
            
            # apply stealth constraints
            current_flat = self._apply_stealth_constraints(current_flat, target_flat, best_params)
            
            # calc loss
            loss = self._calculate_attack_loss(current_flat, target_flat, benign_models)
            
            if loss < best_loss:
                best_loss = loss
                best_params = current_flat.clone()
            
            if iteration > 0 and abs(prev_loss - loss) < self.convergence_threshold:
                break
            
            prev_loss = loss
            
            if iteration % 20 == 0 and iteration > 0:
                self.learning_rate *= 0.9
        
        # apply best parameters back to model
        self._unflatten_model(best_params, model)
        
        return best_loss < float('inf')
    
    def _calculate_similarity_gradient(self, current_params, target_params, attack_direction):
        """Calculate gradient for similarity-based objective."""
        if self.similarity_type == 'cosine':
            return self._cosine_similarity_gradient(current_params, target_params)
        elif self.similarity_type == 'euclidean':
            return self._euclidean_distance_gradient(current_params, target_params)
        elif self.similarity_type == 'l2_norm':
            return self._l2_norm_gradient(current_params, target_params)
        else:
            return self._cosine_similarity_gradient(current_params, target_params)
    
    def _cosine_similarity_gradient(self, current_params, target_params):
        dot_product = torch.dot(current_params, target_params)
        current_norm = torch.norm(current_params)
        target_norm = torch.norm(target_params)
        
        if current_norm == 0:
            return torch.zeros_like(current_params)
        
        grad_numerator = target_params
        grad_denominator = dot_product * current_params / (current_norm ** 2)
        
        grad = (grad_numerator - grad_denominator) / (target_norm * current_norm)
        
        # For attack: we want to minimize similarity (maximize negative similarity)
        return -grad
    
    def _euclidean_distance_gradient(self, current_params, target_params):
        """Gradient for Euclidean distance objective."""
        # This hsould be ||current - target||^2
        return 2 * (current_params - target_params)
    
    def _l2_norm_gradient(self, current_params, target_params):
        """Gradient for L2 norm ratio objective."""
        current_norm = torch.norm(current_params)
        target_norm = torch.norm(target_params)
        
        if current_norm == 0:
            return torch.zeros_like(current_params)
        
        #  ||current|| / ||target||
        return current_params / (current_norm * target_norm)
    
    def _apply_stealth_constraints(self, current_params, target_params, original_params):
        """Apply constraints to maintain stealth."""
        # Constraint 1: Don't deviate too much from original
        max_deviation = self.stealth_factor * torch.norm(original_params)
        deviation = current_params - original_params
        deviation_norm = torch.norm(deviation)
        
        if deviation_norm > max_deviation:
            deviation = deviation * (max_deviation / deviation_norm)
            current_params = original_params + deviation
        
        # Constraint 2: Maintain reasonable norm ratio
        current_norm = torch.norm(current_params)
        target_norm = torch.norm(target_params)
        max_norm_ratio = 5.0  # Maximum allowed norm ratio
        
        if current_norm > max_norm_ratio * target_norm:
            current_params = current_params * (max_norm_ratio * target_norm / current_norm)
        
        return current_params
    
    def _calculate_attack_loss(self, current_params, target_params, benign_models):
        """Calculate overall attack loss."""
        # first objective: similarity manipulation
        if self.similarity_type == 'cosine':
            dot_product = torch.dot(current_params, target_params)
            current_norm = torch.norm(current_params)
            target_norm = torch.norm(target_params)
            
            if current_norm == 0 or target_norm == 0:
                similarity_loss = 1.0
            else:
                cosine_sim = dot_product / (current_norm * target_norm)
                similarity_loss = (cosine_sim - self.target_similarity) ** 2
        
        elif self.similarity_type == 'euclidean':
            distance = torch.norm(current_params - target_params)
            similarity_loss = distance.item()
        
        else:  # L2 norm ratio
            current_norm = torch.norm(current_params)
            target_norm = torch.norm(target_params)
            norm_ratio = current_norm / target_norm if target_norm != 0 else float('inf')
            similarity_loss = (norm_ratio - 1.0) ** 2
        
        return similarity_loss
    
    def _simple_poisoning(self, model, original_model):
        """Simple poisoning when no reference model is available."""
        poison_scale = self.attack_params.get('simple_poison_scale', -1.0)
        
        for param in model.parameters():
            param.data = param.data * poison_scale
        
        metrics = self.calculate_similarity_metrics(original_model, model)
        
        return {
            'attack_success': True,
            'metrics': metrics,
            'similarity_type': 'simple_scaling'
        }


class AdaptiveFakerAttack(FakerAttack):
    """Enhanced Faker attack with adaptive strategies."""
    
    def __init__(self, attack_params=None):
        super().__init__(attack_params)
        self.multi_target = attack_params.get('multi_target', False)
        self.dynamic_similarity = attack_params.get('dynamic_similarity', False)
        self.evasion_strength = attack_params.get('evasion_strength', 1.0)
    
    def _execute_similarity_attack(self, model, reference_model, benign_models=None):
        """Enhanced attack with adaptive target selection."""
        if self.multi_target and benign_models and len(benign_models) > 1:
            return self._multi_target_attack(model, reference_model, benign_models)
        else:
            return super()._execute_similarity_attack(model, reference_model, benign_models)
    
    def _multi_target_attack(self, model, reference_model, benign_models):
        """Attack targeting multiple benign models simultaneously."""
        current_flat = self._flatten_model(model)
        
        # calc weighted target based on multiple benign models
        target_vectors = [self._flatten_model(bm) for bm in benign_models]
        weights = [1.0 / len(benign_models)] * len(benign_models)
        
        weighted_target = torch.zeros_like(current_flat)
        for i, target_vec in enumerate(target_vectors):
            weighted_target += weights[i] * target_vec
        
        # execute attack with weighted target
        best_loss = float('inf')
        for iteration in range(self.max_iterations):
            # calc gradient towards weighted target
            grad = self._calculate_similarity_gradient(current_flat, weighted_target, None)
            
            # update with adaptive learning rate
            adaptive_lr = self.learning_rate * (1.0 - iteration / self.max_iterations)
            current_flat = current_flat - adaptive_lr * grad
            
            # calc loss against all targets
            total_loss = 0
            for target_vec in target_vectors:
                loss = self._calculate_attack_loss(current_flat, target_vec, None)
                total_loss += loss
            
            if total_loss < best_loss:
                best_loss = total_loss
                best_params = current_flat.clone()
        
        # apply best parameters
        self._unflatten_model(best_params, model)
        return True