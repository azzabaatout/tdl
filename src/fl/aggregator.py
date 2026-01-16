import torch
from collections import OrderedDict
from src.models.model_utils import flatten_model


class Aggregator:
    def __init__(self, aggregation_method='fedavg'):
        self.aggregation_method = aggregation_method

    def aggregate(self, client_models, client_weights=None):
        """Aggregate client models into a global model."""
        if self.aggregation_method == 'fedavg':
            return self._federated_averaging(client_models, client_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def _federated_averaging(self, client_models, client_weights):
        """Standard FedAvg aggregation."""
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # weights normalization
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # agg aprams
        global_state_dict = OrderedDict()
        
        # get the structure from the first model
        for name, param in client_models[0].named_parameters():
            global_state_dict[name] = torch.zeros_like(param)
        
        # weighted sum of all client models
        for i, model in enumerate(client_models):
            for name, param in model.named_parameters():
                global_state_dict[name] += client_weights[i] * param.data

        return global_state_dict


class DefensiveAggregator(Aggregator):
    def __init__(self, defense_method='none', **defense_params):
        super().__init__()
        self.defense_method = defense_method
        self.defense_params = defense_params

    def aggregate(self, client_models, client_weights, server_model=None,
                  client_ids=None, current_round=None, global_model=None):
        """Aggregate with defensive mechanisms."""
        if self.defense_method == 'none':
            return self._federated_averaging(client_models, client_weights)
        elif self.defense_method == 'krum':
            return self._krum_aggregation(client_models, client_ids)
        elif self.defense_method == 'norm_clipping':
            return self._norm_clipping_aggregation(client_models, client_weights, client_ids)
        elif self.defense_method == 'fltrust':
            return self._fltrust_aggregation(client_models, server_model, client_ids, global_model)
        elif self.defense_method in ['spp']:
            return self._spp_aggregation(client_models, server_model, client_ids, current_round, global_model)
        else:
            raise ValueError(f"Unknown defense method: {self.defense_method}")

    def _krum_aggregation(self, client_models, client_ids=None):
        """
        Krum defense: select model(s) with minimum distance sum to neighbors.
        Paper Spec: Sum closest n - f - 2 neighbors.
        """
        n_models = len(client_models)
        # m (or f) is the assumed number of malicious clients
        m = self.defense_params.get('num_malicious', n_models // 4)
        multi_krum = self.defense_params.get('multi_krum', False)
        num_selected = self.defense_params.get('num_selected', 1)

        # calc pairwise distances (flat euclidean)
        distances = torch.zeros(n_models, n_models)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                flat_i = flatten_model(client_models[i])
                flat_j = flatten_model(client_models[j])
                dist = torch.norm(flat_i - flat_j).item()
                distances[i, j] = dist
                distances[j, i] = dist

        # calc Krum scores
        # krum requires summing the smallest (n - m - 2) distances.
        # if n - m - 2 < 1, we default to 1 neighbor (or just 0 if n is too small).
        k = max(1, n_models - m - 2)

        scores = []
        for i in range(n_models):
            # sprt distances for model i. index  0 is self (0.0).
            # we take neighbors 1 to k (inclusive of k).
            # slice is [1 : k+1]
            sorted_distances = torch.sort(distances[i])[0]
            score = torch.sum(sorted_distances[1 : k + 1]).item()
            scores.append(score)

        # stanford anomaly detection logic for logging (mean + 1.5 std)
        score_array = torch.tensor(scores)
        score_mean = torch.mean(score_array)
        score_std = torch.std(score_array)
        threshold = score_mean + 1.5 * score_std if score_std > 0 else score_mean

        detected_malicious = []
        for i, score in enumerate(scores):
            if score > threshold:
                if client_ids is not None:
                    detected_malicious.append(client_ids[i])
                else:
                    detected_malicious.append(i)

        if multi_krum and len(client_models) >= num_selected:
            # Select 'num_selected' lowest scores
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
            selected_indices = sorted_indices[:num_selected]
            selected_models = [client_models[i] for i in selected_indices]

            # Average the selected models (Multi-Krum)
            aggregated_model = OrderedDict()
            for name, param in selected_models[0].named_parameters():
                aggregated_model[name] = torch.zeros_like(param.data)

            for model in selected_models:
                for name, param in model.named_parameters():
                    aggregated_model[name] += param.data / len(selected_models)
        else:
            # Single Krum: select absolute minimum
            selected_idx = scores.index(min(scores))
            selected_model = client_models[selected_idx]
            aggregated_model = OrderedDict(selected_model.named_parameters())
            selected_indices = [selected_idx]

        # Determine rejected clients
        rejected_indices = [i for i in range(n_models) if i not in selected_indices]
        rejected_clients = [client_ids[i] for i in rejected_indices] if client_ids else rejected_indices

        defense_stats = {
            'defense_type': 'krum',
            'detected_malicious': detected_malicious,
            'rejected_clients': rejected_clients,
            'rejected_count': len(rejected_clients),
            'threshold_used': threshold.item() if isinstance(threshold, torch.Tensor) else threshold,
            'scores': scores,
            'total_clients': n_models,
            'selected_count': len(selected_indices)
        }
        self.last_defense_stats = defense_stats

        return aggregated_model, defense_stats

    def _norm_clipping_aggregation(self, client_models, client_weights, client_ids=None):
        """
        Norm clipping defense.
        Paper Check: "Norm-clipping sets an upper bound for the value of L2 norm for each local model."
        """
        max_norm = self.defense_params.get('max_norm', 10.0)

        filtered_models = []
        filtered_weights = []
        rejected_clients = []

        for i, model in enumerate(client_models):
            model_norm = 0
            for param in model.parameters():
                model_norm += torch.norm(param.data) ** 2
            model_norm = model_norm ** 0.5

            if model_norm <= max_norm:
                filtered_models.append(model)
                if client_weights:
                    filtered_weights.append(client_weights[i])
            else:
                if client_ids is not None:
                    rejected_clients.append(client_ids[i])

        # fallback if all rejected
        if not filtered_models:
            filtered_models = [client_models[0]]
            filtered_weights = [1.0]

        self.last_defense_stats = {
            'defense_type': 'norm_clipping',
            'total_clients': len(client_models),
            'rejected_count': len(rejected_clients),
            'rejected_clients': rejected_clients,
            'filtered_count': len(filtered_models),
            'max_norm_threshold': max_norm
        }

        global_state_dict = self._federated_averaging(filtered_models, filtered_weights)
        return global_state_dict, self.last_defense_stats


    def _fltrust_aggregation(self, client_models, server_model, client_ids, global_model):
        """
        FLTrust Aggregation.
        Requires 'server_model' to be a trained update on server data relative to 'global_model'.
        Returns: global_model + weighted_sum(scaled_updates)
        """
        if server_model is None or global_model is None:
            raise ValueError("FLTrust requires server_model and global_model")

        eps = 1e-12

        # 1. comp Server Update (Delta)
        # root_update = server_model - global_model
        root_update = OrderedDict()
        root_sqsum = 0.0
        for (name, p_srv), (_, p_g) in zip(server_model.named_parameters(), global_model.named_parameters()):
            du = (p_srv.data - p_g.data)
            root_update[name] = du
            root_sqsum += torch.sum(du * du).item()
        root_norm = (root_sqsum ** 0.5) + eps

        # 2. comp Trust Scores and Scaled Updates
        trusts = []
        scaled_updates = []

        # flatten root update once for efficiency
        root_flat = torch.cat([u.flatten() for u in root_update.values()])

        for model in client_models:
            # client Update: Delta_i = w_i - w_g
            upd = OrderedDict()
            sqsum = 0.0
            for (name, p_i), (_, p_g) in zip(model.named_parameters(), global_model.named_parameters()):
                du = (p_i.data - p_g.data)
                upd[name] = du
                sqsum += torch.sum(du * du).item()
            upd_norm = (sqsum ** 0.5) + eps

            # cos Similarity - relu
            upd_flat = torch.cat([u.flatten() for u in upd.values()])
            cosine = torch.dot(upd_flat, root_flat) / (torch.norm(upd_flat) * torch.norm(root_flat) + eps)
            t = max(0.0, float(cosine))  # Trust score

            # Scale update: Delta_scaled = Delta_i * (||Delta_root|| / ||Delta_i||)
            scale = root_norm / upd_norm
            for k in upd:
                upd[k] = upd[k] * scale

            trusts.append(t)
            scaled_updates.append(upd)

        # 3. norm trust scores
        trust_sum = sum(trusts)
        if trust_sum <= eps:
            # if no model is trusted, return original global model (no update)
            new_state = OrderedDict((name, p.data.clone()) for name, p in global_model.named_parameters())
            defense_stats = {
                "defense_type": "fltrust",
                "rejected_count": len(client_models),
                "rejected_clients": client_ids if client_ids else list(range(len(client_models)))
            }
            self.last_defense_stats = defense_stats
            return new_state, defense_stats

        weights = [t / trust_sum for t in trusts]

        # 4. aggre: w_new = w_g + sum(weight_i * scaled_update_i)
        new_state = OrderedDict()
        for name, p_g in global_model.named_parameters():
            agg_update = torch.zeros_like(p_g.data)
            for w, upd in zip(weights, scaled_updates):
                agg_update += w * upd[name]
            new_state[name] = p_g.data + agg_update

        # identify rejected clients (Trust score = 0)
        rejected_clients = []
        if client_ids is not None:
            rejected_clients = [cid for cid, t in zip(client_ids, trusts) if t <= 0]

        defense_stats = {
            "defense_type": "fltrust",
            "total_clients": len(client_models),
            "rejected_count": len(rejected_clients),
            "rejected_clients": rejected_clients,
        }
        self.last_defense_stats = defense_stats
        return new_state, defense_stats


    def _spp_aggregation(self, client_models, server_model, client_ids=None, current_round=None, global_model=None):
        """SPP defense wrapper."""
        if server_model is None:
            raise ValueError("Server model required for SPP defense")

        if self.defense_method == 'spp':
            from ..defenses.spp import SPP
            spp_defense = SPP(self.defense_params)
        elif self.defense_method == 'adaptive_spp':
            from ..defenses.spp import AdaptiveSPP
            spp_defense = AdaptiveSPP(self.defense_params)
        else:
            from ..defenses.spp import SPP
            spp_defense = SPP(self.defense_params)

        defense_results = spp_defense.defend(
            client_models=client_models,
            server_model=server_model,
            current_round=current_round,
            global_model=global_model
        )

        filtered_models = defense_results['filtered_models']
        weights = defense_results['weights']

        rejected_clients = []
        if client_ids is not None:
            filtered_indices = []
            for filtered_model in filtered_models:
                for i, original_model in enumerate(client_models):
                    if filtered_model is original_model:
                        filtered_indices.append(i)
                        break
            rejected_clients = [client_ids[i] for i in range(len(client_ids))
                                if i not in filtered_indices]

        defense_stats = {
            'defense_type': self.defense_method,
            'total_clients': len(client_models),
            'rejected_count': defense_results.get('rejected_count', 0),
            'rejected_clients': rejected_clients,
            'filtered_count': len(filtered_models),
            'defense_applied': True,
        }

        global_state_dict = self._federated_averaging(filtered_models, weights)
        return global_state_dict, defense_stats
