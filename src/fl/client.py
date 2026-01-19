import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import numpy as np

from src.attacks.faker import FakerAttack


class FLClient:
    def __init__(self, client_id, model, train_data, test_data=None,
                 learning_rate=0.01, local_epochs=1, batch_size=32):
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        if test_data:
            self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def update_model(self, global_state_dict):
        """Update local model with global parameters."""
        self.model.load_state_dict(global_state_dict)

    def local_train(self):
        """Perform local training for specified epochs."""
        self.model.to(self.device)
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def local_test(self):
        """Evaluate local model on test data."""
        if not self.test_data:
            return 0.0, 0.0

        self.model.to(self.device)
        self.model.eval()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def get_model_parameters(self):
        """Get current model parameters."""
        return copy.deepcopy(self.model.state_dict())

    def get_data_size(self):
        """Get size of training data."""
        return len(self.train_data)


class MaliciousClient(FLClient):
    def __init__(self, client_id, model, train_data, test_data=None,
                 learning_rate=0.01, local_epochs=1, batch_size=32,
                 attack_method='none', attack_params=None):
        super().__init__(client_id, model, train_data, test_data,
                         learning_rate, local_epochs, batch_size)
        self.attack_method = attack_method
        self.attack_params = attack_params or {}
        self.is_malicious = True
        # store reference model state for Faker attack
        self.reference_state_dict = None
        # faker attack will be instantiated per-attack with the local update

        # for cooperative Krum attack (Section 5.4 of the paper):
        # for attacking Krum, we let attacker i send its obtained poisoned
        # local model w̄_i to the other m−1 attackers, who also submit w_i
        self.shared_poisoned_state = None  # will be set by coordinator for Krum
        self.is_lead_attacker = False  # only the lead attacker generates the poisoned model

    def set_shared_poisoned_state(self, state_dict):
        """
        Set shared poisoned model state for cooperative Krum attack.
        Called by the experiment coordinator after lead attacker generates the poisoned model.
        """
        self.shared_poisoned_state = copy.deepcopy(state_dict)

    def set_as_lead_attacker(self, is_lead=True):
        """Mark this client as the lead attacker who generates the poisoned model."""
        self.is_lead_attacker = is_lead

    def update_model(self, global_state_dict):
        """Update local model with global parameters, storing reference for Faker attack."""
        # store the global model state as reference before updating
        self.reference_state_dict = copy.deepcopy(global_state_dict)
        # call parent to update the model
        super().update_model(global_state_dict)

    def local_train(self):
        """Train locally and then apply attack."""
        # first perform normal training
        train_loss = super().local_train()

        # then apply attack to the model
        if self.attack_method != 'none':
            self.apply_attack()

        return train_loss

    def apply_attack(self):
        """Apply the specified attack to the local model."""
        # for cooperative Krum attack: if shared_poisoned_state is set,
        # use it directly instead of generating a new poisoned model.
        # This implements Section 5.4: we let attacker i send its obtained
        # poisoned local model w_i (w tilde) to the other m−1 attackers, who also submit ww_i
        if self.shared_poisoned_state is not None:
            print(f"[DEBUG] Client {self.client_id}: Using shared poisoned model (cooperative Krum)")
            self.model.load_state_dict(self.shared_poisoned_state)
            return

        if self.attack_method == 'la':
            self._local_attack()
        elif self.attack_method == 'mb':
            self._model_replacement_attack()
        elif self.attack_method == 'faker':
            self._faker_attack()
        else:
            pass  # No attack

    def _local_attack(self):
        """Local Attack (LA): Change parameter directions."""
        scaling_factor = self.attack_params.get('scaling_factor', 10.0)

        for param in self.model.parameters():
            # gen random signs
            random_signs = torch.randint(0, 2, param.shape, device=param.device) * 2 - 1
            # scale and flip signs
            param.data = param.data * random_signs * scaling_factor

    def _model_replacement_attack(self):
        """Model Replacement Attack (MB): Replace with malicious model."""
        scaling_factor = self.attack_params.get('scaling_factor', -10.0)

        for param in self.model.parameters():
            param.data = param.data * scaling_factor

    def _faker_attack(self):
        """
        Faker Attack - Defense-aware poisoning from the paper.

        Dispatches to the appropriate Faker variant based on target_defense parameter:
        - faker_krum: Theorem 3 (Euclidean distance constraint)
        - faker_fltrust: Theorem 2 (Cosine similarity constraint)
        - faker_norm_clipping: Theorem 4 (L2 norm constraint)
        """
        target_defense = self.attack_params.get('target_defense', 'krum')

        if self.reference_state_dict is None:
            print(f"[DEBUG] Client {self.client_id}: No reference model, falling back to simple negation")
            for param in self.model.parameters():
                param.data = param.data * -1.0
            return

        if target_defense == 'krum':
            self._faker_krum()
        elif target_defense == 'fltrust':
            self._faker_fltrust()
        elif target_defense == 'norm_clipping':
            self._faker_norm_clipping()
        elif target_defense == 'none':
            self._faker_no_defense()
        else:
            print(f"[DEBUG] Client {self.client_id}: Unknown target defense '{target_defense}', using krum")
            self._faker_krum()

    def _faker_no_defense(self):
        """
        Faker Attack when no defense is present.
        Simply applies maximum adversarial impact without constraints.
        """
        print(f"[DEBUG] Client {self.client_id}: Running Faker-NoDefense (unconstrained)")

        scale_factor = self.attack_params.get('scale_factor', -5.0)
        noise_scale = self.attack_params.get('noise_scale', 0.01)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.reference_state_dict:
                    global_param = self.reference_state_dict[name].to(param.device)
                    # compute update
                    update = param.data - global_param
                    # scale the update (negative = adversarial direction)
                    poisoned_update = update * scale_factor
                    # add noise
                    noise = torch.randn_like(param) * noise_scale * torch.norm(update)
                    # apply
                    param.data = global_param + poisoned_update + noise

        print(f"[DEBUG] Client {self.client_id}: Applied scale_factor = {scale_factor}")

    def _faker_krum(self):
        """
        Faker Attack for Krum using scipy optimization from faker.py.

        Theorem 3 constraint: E(w_i, w_i) < E(wg, wi)
        Working with UPDATES (not full models):
        - update Δ = w_local - w_global
        - poisoned update = α * Δ
        - constraint: ||(α-1)*Δ|| < ||Δ|| → |α-1| < 1 → α ∈ (0, 2)

        This allows meaningful perturbations while satisfying the constraint.
        """
        print(f"[DEBUG] Client {self.client_id}: Running Faker-Krum (scipy optimization)")

        # flatten updates to numpy
        update_flat = []
        param_shapes = {}
        param_names = []

        for name, param in self.model.named_parameters():
            if name in self.reference_state_dict:
                global_param = self.reference_state_dict[name].to(param.device)
                update = param.data - global_param  # this is delta= w_local - w_global
                update_flat.append(update.flatten())
                param_shapes[name] = param.shape
                param_names.append(name)

        w_update = torch.cat(update_flat)
        device = w_update.device

        update_norm = torch.norm(w_update).item()
        if update_norm < 1e-10:
            print(f"[DEBUG] Client {self.client_id}: Update too small, skipping attack")
            return

        # convert to numpy for FakerAttack
        update_np = w_update.cpu().numpy()

        # for Krum with updates:
        # - local_model = update delta
        # - global_model = zeros (reference point for updates)
        # - dist_budget = ||0- Δ|| = ||delta|| (the update norm)
        # - constraint: ||(α-1)*delta|| < ||delta|| means |alpha-1| < 1, so alpha ∈ (0, 2)
        zeros_np = np.zeros_like(update_np)

        # get attack params
        num_groups = self.attack_params.get('num_groups', 10)

        # create faker attack instance with updates
        faker = FakerAttack(
            local_model=update_np,
            defense_type='krum',
            num_groups=num_groups,
            global_model=zeros_np
        )

        # gen poisoned update via scipy optimization
        poisoned_update_np, alpha_opt, metrics = faker.generate_poisoned_model(use_grouping=True)

        # convert back to torch
        poisoned_update = torch.from_numpy(poisoned_update_np).float().to(device)

        print(f"[DEBUG] Client {self.client_id}: Optimization success = {metrics.get('success', False)}")
        print(f"[DEBUG] Client {self.client_id}: Update norm ||delta|| = {update_norm:.4f}")
        print(f"[DEBUG] Client {self.client_id}: Euclidean dist ||(α-1)*delta|| = {metrics.get('euclidean_distance', 0):.4f}")
        print(f"[DEBUG] Client {self.client_id}: Difference deltai = {metrics.get('difference', 0):.4f}")

        # Verify constraint
        actual_dist = metrics.get('euclidean_distance', 0)
        print(f"[DEBUG] Client {self.client_id}: Constraint ||(alpha-1)*delta|| < ||delta||: {actual_dist:.4f} < {update_norm:.4f} = {actual_dist < update_norm}")

        # apply poisoned update to model: w_global + poisoned_update
        global_flat = torch.cat([self.reference_state_dict[name].flatten().to(device)
                                 for name in param_names])
        poisoned_flat = global_flat + poisoned_update

        idx = 0
        for name in param_names:
            shape = param_shapes[name]
            numel = int(torch.prod(torch.tensor(shape)).item())
            param_data = poisoned_flat[idx:idx+numel].reshape(shape)

            for param_name, param in self.model.named_parameters():
                if param_name == name:
                    param.data = param_data
                    break
            idx += numel

        print(f"[DEBUG] Client {self.client_id}: Attack complete")

    def _faker_fltrust(self):
        """
        Faker Attack for FLTrust using scipy optimization from faker.py.
        """
        print(f"[DEBUG] Client {self.client_id}: Running Faker-FLTrust (scipy optimization)")
        if self.reference_state_dict is None:
            for param in self.model.parameters():
                param.data = param.data * -1.0
            return

        update_flat = []
        param_shapes = {}
        param_names = []

        for name, param in self.model.named_parameters():
            if name in self.reference_state_dict:
                global_param = self.reference_state_dict[name].to(param.device)
                update = param.data - global_param
                update_flat.append(update.flatten())
                param_shapes[name] = param.shape
                param_names.append(name)

        w_local = torch.cat(update_flat)
        device = w_local.device

        if torch.norm(w_local) < 1e-10:
            return

        local_np = w_local.cpu().numpy()

        num_groups = self.attack_params.get('num_groups', 10)

        # create Faker Attack instance with the local update
        faker = FakerAttack(
            local_model=local_np,
            defense_type='fltrust',
            num_groups=num_groups
        )

        # gen poisoned model via scipy optimization
        poisoned_np, alpha_opt, metrics = faker.generate_poisoned_model(use_grouping=True)

        # comvert back to torch
        poisoned_update = torch.from_numpy(poisoned_np).float().to(device)

        # compute cosine for logging
        cosine_sim = metrics.get('cosine', 0.0)

        print(f"[DEBUG] Client {self.client_id}: Optimization success = {metrics.get('success', False)}")
        print(f"[DEBUG] Client {self.client_id}: Cosine sim = {cosine_sim:.4f}")
        print(f"[DEBUG] Client {self.client_id}: Similarity si = {metrics.get('similarity', 0):.4f}")
        print(f"[DEBUG] Client {self.client_id}: Difference Δi = {metrics.get('difference', 0):.4f}")
        print(f"[DEBUG] Client {self.client_id}: Objective f(α) = {metrics.get('objective', 0):.4f}")

        # apply to model: global + poisoned_update
        global_flat = torch.cat([self.reference_state_dict[name].flatten().to(device)
                                 for name in param_names])
        poisoned_flat = global_flat + poisoned_update

        idx = 0
        for name in param_names:
            shape = param_shapes[name]
            numel = int(torch.prod(torch.tensor(shape)).item())
            param_data = poisoned_flat[idx:idx+numel].reshape(shape)

            for param_name, param in self.model.named_parameters():
                if param_name == name:
                    param.data = param_data
                    break
            idx += numel

        print(f"[DEBUG] Client {self.client_id}: Attack complete (cosine={cosine_sim:.4f})")


    def _optimize_alpha_theorem2(self, alpha, w, w_root, num_groups):
        """
        Optimize α using Theorem 2 closed-form solution.

        For each α_j, given α_{-j} (all others fixed):

        λ = Σ_{k≠j} α_k × w_k²
        β = Σ_{k≠j} α_k² × w_k²
        γ = Σ_{k≠j} α_k

        Optimal α_j = [w_j² × (β - λγ) + sqrt(w_j² × (λ² + w_j²β) × (w_j²γ² + β))]
                      / [w_j² × (λ + w_j²γ)]
        """
        J = len(alpha)
        new_alpha = alpha.clone()

        # use grouped optimization for efficiency (T groups)
        group_size = max(1, J // num_groups)

        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min((g + 1) * group_size, J)

            # compute alpha lambda and gamma excluding current group
            mask = torch.ones(J, dtype=torch.bool, device=alpha.device)
            mask[start_idx:end_idx] = False

            alpha_minus = alpha[mask]
            w_minus = w[mask]

            if len(alpha_minus) == 0:
                continue

            lambda_val = torch.sum(alpha_minus * w_minus ** 2)
            beta_val = torch.sum(alpha_minus ** 2 * w_minus ** 2)
            gamma_val = torch.sum(alpha_minus)

            # optimize each alpha j  in this group
            for j in range(start_idx, end_idx):
                w_j = w[j]
                w_j_sq = w_j ** 2

                # handle edge case where w j  approx 0
                if abs(w_j) < 1e-10:
                    new_alpha[j] = 1.0
                    continue

                # theo 2 formula
                discriminant = w_j_sq * (lambda_val ** 2 + w_j_sq * beta_val) * (w_j_sq * gamma_val ** 2 + beta_val)

                if discriminant < 0:
                    # fallback to positive value
                    new_alpha[j] = 1.0
                    continue

                numerator = w_j_sq * (beta_val - lambda_val * gamma_val) + torch.sqrt(discriminant)
                denominator = w_j_sq * (lambda_val + w_j_sq * gamma_val)

                if abs(denominator) < 1e-10:
                    new_alpha[j] = 1.0
                else:
                    optimal_alpha_j = numerator / denominator
                    new_alpha[j] = torch.clamp(optimal_alpha_j, min=0.1, max=10.0)

        return new_alpha

    def _cosine_similarity_vec(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        dot_product = torch.dot(vec1, vec2)
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return (dot_product / (norm1 * norm2)).item()

    def _faker_norm_clipping(self):
        """
        Faker Attack for Norm-clipping using scipy optimization from faker.py.
        """
        print(f"[DEBUG] Client {self.client_id}: Running Faker-NormClipping (scipy optimization)")

        update_flat = []
        param_shapes = {}
        param_names = []

        for name, param in self.model.named_parameters():
            if name in self.reference_state_dict:
                global_param = self.reference_state_dict[name].to(param.device)
                update = param.data - global_param
                update_flat.append(update.flatten())
                param_shapes[name] = param.shape
                param_names.append(name)

        w_local = torch.cat(update_flat)
        device = w_local.device

        if torch.norm(w_local) < 1e-10:
            return

        # conv to numpy for FakerAttack
        local_np = w_local.cpu().numpy()

        # get attack params
        num_groups = self.attack_params.get('num_groups', 10)

        #  FakerAttack instance with the local update
        faker = FakerAttack(
            local_model=local_np,
            defense_type='norm_clipping',
            num_groups=num_groups
        )

        # gen poisoned model via scipy optimization
        poisoned_np, alpha_opt, metrics = faker.generate_poisoned_model(use_grouping=True)

        # convert back to torch
        poisoned_update = torch.from_numpy(poisoned_np).float().to(device)

        print(f"[DEBUG] Client {self.client_id}: Optimization success = {metrics.get('success', False)}")
        print(f"[DEBUG] Client {self.client_id}: Similarity si = {metrics.get('similarity', 0):.4f}")
        print(f"[DEBUG] Client {self.client_id}: Difference Δi = {metrics.get('difference', 0):.4f}")
        print(f"[DEBUG] Client {self.client_id}: Objective f(α) = {metrics.get('objective', 0):.4f}")

        # apply to model: global + poisoned_update
        global_flat = torch.cat([self.reference_state_dict[name].flatten().to(device)
                                 for name in param_names])
        poisoned_flat = global_flat + poisoned_update

        idx = 0
        for name in param_names:
            shape = param_shapes[name]
            numel = int(torch.prod(torch.tensor(shape)).item())
            param_data = poisoned_flat[idx:idx+numel].reshape(shape)

            for param_name, param in self.model.named_parameters():
                if param_name == name:
                    param.data = param_data
                    break
            idx += numel

        print(f"[DEBUG] Client {self.client_id}: Attack complete")

    def _compute_model_distance(self, model, state_dict):
        """Compute Euclidean distance between model and a state dict."""
        total_dist_sq = 0.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in state_dict:
                    ref_param = state_dict[name]
                    if isinstance(ref_param, torch.Tensor):
                        ref_param = ref_param.to(param.device)
                    diff = param.data - ref_param
                    total_dist_sq += torch.sum(diff ** 2).item()
        return total_dist_sq ** 0.5

    def _compute_model_distance_from_state(self, state_dict1, state_dict2):
        """Compute Euclidean distance between two state dicts."""
        total_dist_sq = 0.0
        for name in state_dict1:
            if name in state_dict2:
                param1 = state_dict1[name]
                param2 = state_dict2[name]
                if isinstance(param1, torch.Tensor) and isinstance(param2, torch.Tensor):
                    diff = param1 - param2.to(param1.device)
                    total_dist_sq += torch.sum(diff ** 2).item()
        return total_dist_sq ** 0.5