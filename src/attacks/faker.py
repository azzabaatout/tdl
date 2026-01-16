import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from typing import Tuple, Optional

class FakerAttack:
    """
    Faker attack implementation based on the paper.
    Generates poisoned models by scaling local model parameters.
    note: refactored and simplified based on the pseudocode available
    """

    def __init__(self, local_model: np.ndarray, defense_type: str = 'fltrust', num_groups: int = 10):
        self.local_model = np.array(local_model).flatten()
        self.J = len(self.local_model)
        self.num_groups = min(num_groups, self.J)
        self.group_size = max(1, self.J // self.num_groups)
        self.defense_type = defense_type

    def generate(self, wg_prev: np.ndarray = None):
        if self.defense_type == 'fltrust':
            return self._attack_fltrust()
        elif self.defense_type == 'norm_clipping':
            return self._attack_norm_clipping()
        elif self.defense_type == 'krum':
            return self._attack_krum(wg_prev)
        else:
            raise ValueError(f"Defense {self.defense_type} not supported.")

    def expand_alpha(self, alpha_grouped: np.ndarray) -> np.ndarray:
        """Expand T grouped scalars to J individual scalars."""
        alpha_full = np.repeat(alpha_grouped, self.group_size)
        if len(alpha_full) < self.J:
            alpha_full = np.append(alpha_full, [alpha_grouped[-1]]*(self.J - len(alpha_full)))
        return alpha_full[:self.J]

    @staticmethod
    def _cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot / (norm1 * norm2 + 1e-10)

    def compute_similarity(self, alpha: np.ndarray) -> float:
        """Compute defense-specific similarity metric."""
        poisoned = alpha * self.local_model

        if self.defense_type == 'fltrust':
            cos_sim = self._cosine_sim(poisoned, self.local_model)
            norm_ratio = np.linalg.norm(poisoned) / (np.linalg.norm(self.local_model) + 1e-10)
            return cos_sim * norm_ratio

        elif self.defense_type == 'krum':
            euclidean_dist = np.linalg.norm(poisoned - self.local_model)
            return -euclidean_dist

        elif self.defense_type == 'norm_clipping':
            return np.linalg.norm(poisoned)

        else:
            return self._cosine_sim(poisoned, self.local_model)

    def compute_difference(self, alpha: np.ndarray) -> float:
        """Compute Δi = sum of |alpha_j - 1|."""
        return np.sum(np.abs(alpha - 1.0))

    def get_similarity_bounds(self) -> Tuple[float, float]:
        """Get defense-specific similarity bounds."""
        if self.defense_type == 'fltrust':
            return 0.0, 1e9  # Cosine >= 0
        elif self.defense_type == 'norm_clipping':
            benign_norm = np.linalg.norm(self.local_model)
            return -1e9, benign_norm
        elif self.defense_type == 'krum':
            return -1e9, 0.0  # Euclidean dist <= 0 (similarity is negative dist)
        else:
            return 0.0, 1.0

    def objective_function(self, alpha: np.ndarray) -> float:
        """Maximize f(α) = si * Δi (return negative for minimization)."""
        si = self.compute_similarity(alpha)
        di = self.compute_difference(alpha)
        return -(si * di)

    def _attack_fltrust(self):
        # Theorem 2: Maximize Si * Di.
        # analyzical solution sets J-1 scalars and solves for the last.
        # heuristic (might not be accuarte): Large positive scaling that maintains direction (cos > 0).
        alpha = np.random.uniform(1.5, 3.0, self.J)
        # ensure cosine similarity remains positive (C1)
        poisoned = alpha * self.wi
        return poisoned

    def _attack_norm_clipping(self):
        # theorem 4: Constraint ||wi'|| <= ||wi||
        # we maximizesum|alpha-1| by varying scalars around 1.0
        alpha = np.ones(self.J)
        perturb = np.random.uniform(0.1, 0.8, self.J // 2)
        alpha[:self.J//2] += perturb
        alpha[self.J//2:2*(self.J//2)] -= perturb

        current_norm = np.linalg.norm(alpha * self.wi)
        benign_norm = np.linalg.norm(self.wi)
        alpha *= (benign_norm / (current_norm + 1e-10))
        return alpha * self.wi

    def _attack_krum(self, wg_prev):
        # Theorem 3: E(wi', wi) <= E(wg_prev, wi)
        if wg_prev is None:
            raise ValueError("Krum attack requires previous global model wg.")

        dist_budget = np.linalg.norm(wg_prev.flatten() - self.wi)
        # scale parameters to the edge of the Euclidean distance budget
        scaling_factor = 1.0 + (dist_budget / (np.linalg.norm(self.wi) + 1e-10))
        return self.wi * scaling_factor

    def generate_poisoned_model(self, initial_alpha: Optional[np.ndarray] = None,
                                use_grouping: bool = True):
        """Generate poisoned model via optimization."""

        if use_grouping and self.num_groups < self.J:
            if initial_alpha is None:
                # start near 1.0 with small perturbations
                alpha_init = np.ones(self.num_groups) + np.random.randn(self.num_groups) * 0.3
                alpha_init = np.clip(alpha_init, 0.5, 2.0)
            else:
                alpha_init = np.array(initial_alpha)[:self.num_groups]

            if self.defense_type == 'fltrust':
                def similarity_constraint(alpha_grouped):
                    alpha_full = self.expand_alpha(alpha_grouped)
                    poisoned = alpha_full * self.local_model
                    cos = self._cosine_sim(poisoned, self.local_model)
                    return cos

                sl, su = 0.01, 0.7
            else:
                def similarity_constraint(alpha_grouped):
                    alpha_full = self.expand_alpha(alpha_grouped)
                    return self.compute_similarity(alpha_full)

                sl, su = self.get_similarity_bounds()

            constraint = NonlinearConstraint(similarity_constraint, lb=sl, ub=su)
            bounds = [(0.5, 2.0)] * self.num_groups  # a tighter bound

            def objective(alpha_grouped):
                alpha_full = self.expand_alpha(alpha_grouped)
                return self.objective_function(alpha_full)
            #  this is more computationally intensive but theoretically
            #  more accurate than the paper's "fast" heuristic
            # (no other choice)
            result = minimize(
                objective,
                x0=alpha_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint,
                options={'maxiter': 500, 'ftol': 1e-8}
            )

            alpha_opt = self.expand_alpha(result.x)

        else:
            # similar for non-grouped...
            pass

        poisoned_model = alpha_opt * self.local_model

        # compute final metrics
        if self.defense_type == 'fltrust':
            final_cos = self._cosine_sim(poisoned_model, self.local_model)
            final_si = self.compute_similarity(alpha_opt)
        else:
            final_cos = None
            final_si = self.compute_similarity(alpha_opt)

        final_di = self.compute_difference(alpha_opt)


        return poisoned_model, alpha_opt, {
            'success': result.success,
            'cosine': final_cos,
            'similarity': final_si,
            'difference': final_di,
            'objective': final_si * final_di
        }