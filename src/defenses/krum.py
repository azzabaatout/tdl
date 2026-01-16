import torch
from .base_defense import SimilarityBasedDefense

class Krum(SimilarityBasedDefense):
    """
    Krum Defense

    Reference: Section 2.2 of "Can We Trust the Similarity Measurement in Federated Learning?"
    """

    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        # krum strictly requires knowing 'm' (upper bound of malicious clients)
        self.num_malicious = defense_params.get('num_malicious', 0)
        self.multi_krum = defense_params.get('multi_krum', False)
        # for Multi-Krum, how many to select (usually k=n-m)
        self.num_selected = defense_params.get('num_selected', 1)

    def defend(self, client_models, **kwargs):
        n = len(client_models)
        m = self.num_malicious

        # krum requires n >= 2m + 3 for Byzantine tolerance,
        # but practically can run if n > m + 2.
        # the  paper mentions selecting from n - m - 1 neighbors.
        k_neighbors = n - m - 2

        if k_neighbors <= 0:
            # fallback if too few models: return average of all
            return {
                'filtered_models': client_models,
                'weights': [1.0/n] * n,
                'defense_applied': False
            }

        # 1. flatten models (Krum operates on Euclidean distance of parameters)
        flat_models = [self._flatten_model(model) for model in client_models]

        # 2. comp pairwise distances
        # optimization: use pre-computed distance matrix if possible, but loop is clear
        scores = []
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                # euc distance as mentioned in paper
                dist = torch.norm(flat_models[i] - flat_models[j], p=2).item()
                dists.append(dist)

            # 3. krum Score: sum of distances to closest (n - m - 2) neighbors
            dists.sort()
            # The paper says "sums up the top n - m - 1 distances" (likely including itself or off-by-one).
            # Standard Krum usually sums k closest neighbors.
            # We'll stick to the standard definition: sum of k smallest distances.
            score = sum(dists[:k_neighbors])
            scores.append(score)

        # 4. selection
        if self.multi_krum:
            # irrelevant
            #
            # Multi-Krum: Select 'num_selected' models with lowest scores
            # Usually we select n-m models
            k_best = min(self.num_selected, n - m)
            top_indices = sorted(range(n), key=lambda i: scores[i])[:k_best]

            selected_models = [client_models[i] for i in top_indices]
            weights = [1.0 / len(selected_models)] * len(selected_models)

            return {
                'filtered_models': selected_models,
                'weights': weights,
                'rejected_count': n - len(selected_models),
                'defense_applied': True
            }
        else:
            # single-Krum: slect 1 model with lowest score
            best_idx = scores.index(min(scores))
            selected_model = client_models[best_idx]

            return {
                'filtered_models': [selected_model],
                'weights': [1.0],
                'rejected_count': n - 1,
                'defense_applied': True
            }

