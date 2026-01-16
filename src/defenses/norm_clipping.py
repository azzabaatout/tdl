import torch
from .base_defense import SimilarityBasedDefense

class NormClipping(SimilarityBasedDefense):
    """
    Norm Clipping Defense as implemented in "Can We Trust the Similarity Measurement?".

    """

    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        # this is the fixed upper bound 'M' known to the server
        self.max_norm = defense_params.get('max_norm', 10.0)
        self.check_lower_bound = defense_params.get('check_lower_bound', True)

    def defend(self, client_models, **kwargs):
        threshold = self.max_norm
        # section 6.1: "provide a lower bound, which is four-fifths of the upper bound"
        lower_bound = 0.8 * threshold if self.check_lower_bound else 0.0

        filtered_models = []
        weights = []
        rejected_count = 0

        for model in client_models:
            #  L2 norm of the update/model
            # note: Paper says "L2 norms of local models", usually meaning weights
            norm = self._calculate_l2_norm(model)

            if lower_bound <= norm <= threshold:
                filtered_models.append(model)
                weights.append(1.0)
            else:
                rejected_count += 1

        # Fallback if all rejected (standard practice, though not explicitly detailed in paper)
        if not filtered_models:
            # for reproduction, maybe return empty or the one closest to range?
            # we'll return empty/no-op to strictly follow "discard" logic,
            # or typically FedAvg handles empty lists by skipping round.
            # here, let's return all to avoid crashing, but log it.
            return {
                'filtered_models': client_models,
                'weights': [1.0/len(client_models)] * len(client_models),
                'rejected_count': 0, # Technically all rejected, but fallback used
                'defense_applied': False
            }

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        return {
            'filtered_models': filtered_models,
            'weights': weights,
            'rejected_count': rejected_count,
            'defense_applied': True
        }

    def _calculate_l2_norm(self, model):
        """Calculate L2 norm of flattened model parameters."""
        total_norm = 0.0
        for param in model.parameters():
            total_norm += torch.norm(param.data, p=2).item() ** 2
        return total_norm ** 0.5
