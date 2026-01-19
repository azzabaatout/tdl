import torch
import random
from .base_defense import SimilarityBasedDefense


class SPP(SimilarityBasedDefense):
    """
    Similarity of Partial Parameters (SPP) Defense.

    From Section 7 of the paper "Can We Trust the Similarity Measurement in FL":
    1. Receive submissions: Collect all local model updates from clients
    2. Generate random key: Select random subset of parameter indices J' ⊂ {1,...,J}
       (typically J/2 parameters)
    3. Extract partial vectors: For each client model w̃i and reference model wr,
       extract values at indices J' to create partial vectors
    4. Evaluate similarity: Calculate similarity S(ṽi, vr) on partial vectors
    5. Filter/Aggregate: Discard models that fail the partial similarity test


    """

    def __init__(self, defense_params=None):
        super().__init__(defense_params)
        self.selection_ratio = defense_params.get('selection_ratio', 0.5)  # J/2 as per paper
        self.current_round = 0
        self.cosine_threshold = defense_params.get('cosine_threshold', 0.9)

        print(f"[SPP INIT] cosine_threshold = {self.cosine_threshold}")
        print(f"[SPP INIT] selection_ratio = {self.selection_ratio}")

    def defend(self, client_models, server_model=None, current_round=None, global_model=None, **kwargs):
        """
        SPP defense: Compare MODEL PARAMETERS (not updates) on random partial indices.

        As per paper Section 7:
        - Extract partial vectors directly from client models and reference model
        - Compute similarity on these partial vectors
        - Filter models that fail similarity threshold
        """
        if current_round is not None:
            self.current_round = current_round
        else:
            self.current_round += 1

        print(f"\n[SPP DEBUG] Round {self.current_round}, Threshold: {self.cosine_threshold}")
        print(f"[SPP DEBUG] Evaluating {len(client_models)} client models")
        print(f"[SPP DEBUG] Server model present: {server_model is not None}")

        # use server_model as reference model (wr in paper notation)
        # if not available, cannot perform SPP
        if server_model is None:
            print("[SPP DEBUG] WARNING: No server/reference model - cannot compute similarity!")
            return {'filtered_models': client_models, 'weights': [1/len(client_models)]*len(client_models)}

        reference_model = server_model

        # 1. flatten reference model to get total parameter count
        reference_flat = self._flatten_model(reference_model)
        total_params = reference_flat.numel()

        # 2. gen random indices for this round (J' ⊂ {1,...,J})
        num_selected = int(total_params * self.selection_ratio)
        # seed with round number so all clients evaluated on SAME subset
        rng = random.Random(self.current_round)
        selected_indices = torch.tensor(rng.sample(range(total_params), num_selected))

        # 3. extract partial vector from reference model
        reference_partial = reference_flat[selected_indices]

        valid_models = []
        weights = []
        rejected_count = 0
        rejected_clients = []

        for idx, model in enumerate(client_models):
            # 4. flatten client model and extract partial vector
            client_flat = self._flatten_model(model)
            client_partial = client_flat[selected_indices]

            # 5. compute cosine similarity on PARTIAL vectors (not updates!)
            sim = self.compute_cosine_similarity(client_partial, reference_partial)

            print(f"[SPP DEBUG] Client {idx}: partial_cosine={sim:.6f}, threshold={self.cosine_threshold}, PASS={sim > self.cosine_threshold}")

            if sim > self.cosine_threshold:
                valid_models.append(model)
                weights.append(sim if sim > 0 else 0)
            else:
                rejected_count += 1
                rejected_clients.append(idx)
                print(f"[SPP DEBUG] >>> REJECTED client {idx}")

        # 6. norm weights and aggregate remaining models
        if not valid_models:
            print("[SPP DEBUG] WARNING: All clients rejected, falling back to FedAvg")
            return {
                'filtered_models': client_models,
                'weights': [1/len(client_models)]*len(client_models),
                'rejected_count': rejected_count,
                'rejected_clients': rejected_clients
            }

        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(valid_models)] * len(valid_models)

        print(f"[SPP DEBUG] Results: {len(valid_models)} accepted, {rejected_count} rejected")

        return {
            'filtered_models': valid_models,
            'weights': weights,
            'rejected_count': rejected_count,
            'rejected_clients': rejected_clients
        }