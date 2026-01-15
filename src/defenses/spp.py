import torch
import random
from .base_defense import SimilarityBasedDefense

class SPP(SimilarityBasedDefense):
    """
    Similarity of Partial Parameters (SPP) Defense.

    Implements the strategy described in Section 7 of the paper:
    evaluating similarity metrics on a random subset of parameters to
    break Faker's optimization.
    """

    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.selection_ratio = defense_params.get('selection_ratio', 0.5)  # Typically J/2
        self.current_round = 0

        # the paper experiments often use SPP with another defense (e.g., "FLTrust w. SPP").
        # iff used standalone, we need a basic rejection policy.
        self.cosine_threshold = defense_params.get('cosine_threshold', 0.0)

    def defend(self, client_models, server_model=None, current_round=None, **kwargs):
        if current_round is not None:
            self.current_round = current_round
        else:
            self.current_round += 1

        # 1. select random indices (fixed for this round)
        # using the server model to determine size is safe
        ref_model = server_model if server_model else client_models[0]
        flat_ref = self._flatten_model(ref_model)
        total_params = flat_ref.numel()
        num_selected = int(total_params * self.selection_ratio)

        # seed ensures all clients in this round are evaluated on SAME subset
        rng = random.Random(self.current_round)
        selected_indices = torch.tensor(rng.sample(range(total_params), num_selected))

        # 2. compute Partial Similarities
        valid_models = []
        weights = []

        # we need a reference vector (e.g. from clean server model)
        if server_model is None:
            # without a reference, SPP cannot compute similarity.
            # fallbacl to FedAvg (or raise error)
            return {'filtered_models': client_models, 'weights': [1/len(client_models)]*len(client_models)}

        server_flat = self._flatten_model(server_model)
        server_partial = server_flat[selected_indices]

        for model in client_models:
            client_flat = self._flatten_model(model)
            client_partial = client_flat[selected_indices]

            # comp similarity on PARTIAL vectors
            sim = self.compute_cosine_similarity(client_partial, server_partial)

            # basic Filtering (Paper doesn t specify complex outlier detection for SPP itself)
            # it implies using the underlying defense's logic but on partial params.
            # her we implement a simple relu like threshold common in fltrust.
            if sim > self.cosine_threshold:
                valid_models.append(model)
                weights.append(sim if sim > 0 else 0)

        # Normalization
        if not valid_models:
            return {'filtered_models': client_models, 'weights': [1/len(client_models)]*len(client_models)}

        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(valid_models)] * len(valid_models)

        return {
            'filtered_models': valid_models,
            'weights': weights,
            'rejected_count': len(client_models) - len(valid_models)
        }
