import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from .base_defense import SimilarityBasedDefense


class SPP(SimilarityBasedDefense):
    """
    Similarity of Partial Parameters (SPP) Defense

    Randomly selects a subset of parameters (typically J/2 out of J total)
    for similarity evaluation instead of evaluating all parameters.
    """

    def __init__(self, defense_params=None):
        super().__init__(defense_params)

        self.selection_ratio = defense_params.get('selection_ratio', 0.5)
        # cos similarity threshold for filtering
        self.cosine_threshold = defense_params.get('cosine_threshold', 0.0)
        # either use ReLU on cosine similarity
        self.use_relu = defense_params.get('use_relu', True)
        # either train server model or use provided one
        self.server_epochs = defense_params.get('server_epochs', 5)
        self.server_lr = defense_params.get('server_lr', 0.01)
        # random seed for parameter selection (changes each round)
        self.current_round = 0

    def defend(self, client_models, server_data=None, global_model=None,
               server_model=None, current_round=None, **kwargs):
        """
        Apply SPP defense mechanism.

        Args:
            client_models: List of client models
            server_data: Clean dataset on server (optional if server_model provided)
            global_model: Current global model (optional if server_model provided)
            server_model: Pre-trained server reference model (optional)
            current_round: Current training round (for varying parameter selection)

        Returns:
            dict: Defense results with filtered models and weights
        """
        # update round counter for varying parameter selection
        if current_round is not None:
            self.current_round = current_round
        else:
            self.current_round += 1

        # step 1: get or train server model as reference
        if server_model is not None:
            reference_model = server_model
        elif server_data is not None and global_model is not None:
            reference_model = self._train_server_model(global_model, server_data)
        else:
            # if no server data or model available, fall back to simple averaging
            return {
                'filtered_models': client_models,
                'weights': [1.0 / len(client_models)] * len(client_models),
                'rejected_count': 0,
                'defense_applied': False
            }

        # step 2: random select partial parameters for this round
        selected_indices = self._select_partial_parameters(reference_model)

        # step 3: eval client models using partial parameters only
        model_scores = self._evaluate_client_models_partial(
            client_models, reference_model, selected_indices
        )

        # step 4: filter and weight client models based on partial similarity
        filtered_results = self._filter_and_weight_models(client_models, model_scores)

        filtered_results['selected_param_count'] = len(selected_indices)
        filtered_results['selection_ratio'] = self.selection_ratio
        filtered_results['current_round'] = self.current_round

        return filtered_results

    def _select_partial_parameters(self, reference_model):
        """
        Randomly select J' parameters out of J total parameters.
        Selection varies across rounds to prevent attacker adaptation.

        Args:
            reference_model: Model to determine parameter structure

        Returns:
            list: Indices of selected parameters
        """
        flat_params = self._flatten_model(reference_model)
        total_params = flat_params.numel()

        # calc number of parameters to select (J')
        num_selected = int(total_params * self.selection_ratio)
        num_selected = max(1, num_selected)

        # use round number as seed for reproducible but varying selection
        # ensures selection changes across rounds
        random.seed(self.current_round)
        torch.manual_seed(self.current_round)

        # randomly select parameter indices
        selected_indices = random.sample(range(total_params), num_selected)
        selected_indices.sort()

        return selected_indices

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

    def _evaluate_client_models_partial(self, client_models, server_model, selected_indices):
        """
        Evaluate client models against server model using only partial parameters.

        Args:
            client_models: List of client models
            server_model: Reference server model
            selected_indices: Indices of parameters to evaluate

        Returns:
            list: Model scores based on partial similarity
        """
        server_flat = self._flatten_model(server_model)
        server_partial = server_flat[selected_indices]

        model_scores = []

        for i, client_model in enumerate(client_models):
            client_flat = self._flatten_model(client_model)
            client_partial = client_flat[selected_indices]

            # calc cosine similarity on partial parameters only
            cosine_sim = self._calculate_cosine_similarity(client_partial, server_partial)

            # apply relu to filter out opposite directions
            if self.use_relu:
                cosine_score = max(0.0, cosine_sim)
            else:
                cosine_score = cosine_sim

            # calc l2 norm ratio on partial parameters
            client_norm = torch.norm(client_partial, p=2)
            server_norm = torch.norm(server_partial, p=2)

            if server_norm != 0:
                norm_ratio = (client_norm / server_norm).item()
            else:
                norm_ratio = float('inf')

            # calc eucl distance on partial parameters
            euclidean_dist = torch.norm(client_partial - server_partial, p=2).item()

            model_scores.append({
                'model_idx': i,
                'cosine_similarity': cosine_sim,
                'cosine_score': cosine_score,
                'norm_ratio': norm_ratio,
                'euclidean_distance': euclidean_dist,
                'final_score': cosine_score
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
        """
        Filter malicious models and compute weights based on partial similarity scores.

        Models that fail the partial similarity check are discarded.
        """
        if len(model_scores) <= 2:
            # not enough models for statistical analysis, use basic filtering
            return self._basic_filter(client_models, model_scores)

        # enhanced filtering with multiple criteria on partial parameters
        valid_models = []
        valid_scores = []
        rejected_count = 0

        # calc statistics for dynamic thresholding
        cosine_scores = [score['cosine_score'] for score in model_scores]
        norm_ratios = [score['norm_ratio'] for score in model_scores]

        cosine_mean = sum(cosine_scores) / len(cosine_scores)
        cosine_std = (sum([(x - cosine_mean) ** 2 for x in cosine_scores]) / len(cosine_scores)) ** 0.5

        norm_mean = sum(norm_ratios) / len(norm_ratios)
        norm_std = (sum([(x - norm_mean) ** 2 for x in norm_ratios]) / len(norm_ratios)) ** 0.5

        # dynamic threshold based on distribution
        if cosine_std > 0:
            dynamic_cosine_threshold = max(self.cosine_threshold, cosine_mean - 1.5 * cosine_std)
        else:
            dynamic_cosine_threshold = self.cosine_threshold

        # apply multi-criteria filtering
        for score_info in model_scores:
            is_valid = True

            # crit 1: cos similarity threshold (on partial parameters)
            if score_info['cosine_score'] < dynamic_cosine_threshold:
                is_valid = False

            # crit 2: stat outlier detection
            if cosine_std > 0:
                cosine_z_score = abs(score_info['cosine_score'] - cosine_mean) / cosine_std
                if cosine_z_score > 2.0:
                    is_valid = False

            # crit 3: norm ratio analysis (on partial parameters)
            if norm_std > 0:
                norm_z_score = abs(score_info['norm_ratio'] - norm_mean) / norm_std
                if norm_z_score > 2.5:  # Very abnormal norm ratio
                    is_valid = False

            # crit 4: very low cosine similarity (potential adversarial)
            if score_info['cosine_score'] < 0.1:
                is_valid = False

            if is_valid:
                valid_models.append(client_models[score_info['model_idx']])
                valid_scores.append(score_info)
            else:
                rejected_count += 1

        if not valid_models:
            sorted_scores = sorted(model_scores, key=lambda x: x['cosine_score'], reverse=True)
            top_half = sorted_scores[:max(1, len(sorted_scores)//2)]

            valid_models = [client_models[score['model_idx']] for score in top_half]
            valid_scores = top_half
            rejected_count = len(model_scores) - len(valid_models)

        # calc weights based on cosine scores
        scores = [score_info['final_score'] for score_info in valid_scores]
        total_score = sum(scores)

        if total_score == 0:
            weights = [1.0 / len(valid_models)] * len(valid_models)
        else:
            weights = [score / total_score for score in scores]

        return {
            'filtered_models': valid_models,
            'weights': weights,
            'rejected_count': rejected_count,
            'defense_applied': True,
            'model_scores': valid_scores,
            'dynamic_threshold': dynamic_cosine_threshold
        }

    def _basic_filter(self, client_models, model_scores):
        """Basic filtering for small numbers of models."""
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
            return {
                'filtered_models': client_models,
                'weights': [1.0 / len(client_models)] * len(client_models),
                'rejected_count': 0,
                'defense_applied': True,
                'all_rejected': True
            }

        scores = [score_info['final_score'] for score_info in valid_scores]
        total_score = sum(scores)

        if total_score == 0:
            weights = [1.0 / len(valid_models)] * len(valid_models)
        else:
            weights = [score / total_score for score in scores]

        return {
            'filtered_models': valid_models,
            'weights': weights,
            'rejected_count': rejected_count,
            'defense_applied': True,
            'model_scores': valid_scores
        }


class AdaptiveSPP(SPP):
    """
    Adaptive SPP that dynamically adjusts the selection ratio based on attack detection.

    If many models are rejected, it may increase the selection ratio to be more thorough.
    If few models are rejected, it may decrease the ratio for efficiency.
    """

    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.min_selection_ratio = defense_params.get('min_selection_ratio', 0.25)
        self.max_selection_ratio = defense_params.get('max_selection_ratio', 0.75)
        self.adaptive_adjustment = defense_params.get('adaptive_adjustment', True)
        self.rejection_history = []

    def defend(self, client_models, **kwargs):
        """Apply adaptive SPP defense with dynamic parameter selection."""
        # Adjust selection ratio based on recent rejection rates
        if self.adaptive_adjustment and len(self.rejection_history) >= 3:
            self._adjust_selection_ratio()

        # apply standard SPP defense
        results = super().defend(client_models, **kwargs)

        # track rejection rate for adaptive adjustment
        rejection_rate = results['rejected_count'] / len(client_models)
        self.rejection_history.append(rejection_rate)

        # keep only recent history
        if len(self.rejection_history) > 10:
            self.rejection_history.pop(0)

        results['adaptive_selection_ratio'] = self.selection_ratio

        return results

    def _adjust_selection_ratio(self):
        """Dynamically adjust selection ratio based on recent rejection rates."""
        avg_rejection_rate = sum(self.rejection_history[-3:]) / 3

        # if high rejection rate, increase selection ratio (be more thorough)
        if avg_rejection_rate > 0.3:
            self.selection_ratio = min(self.selection_ratio * 1.1, self.max_selection_ratio)
        # of low rejection rate, decrease selection ratio (be more efficient)
        elif avg_rejection_rate < 0.1:
            self.selection_ratio = max(self.selection_ratio * 0.9, self.min_selection_ratio)
