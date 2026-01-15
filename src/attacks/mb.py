import torch
import copy
import numpy as np
from .base_attack import ModelPoisoningAttack

class ManipulateByzantineAttack(ModelPoisoningAttack):
    """
    Manipulate Byzantine (MB) Attack

    Finds a single shared scalar 'lambda' such that w_poison = lambda * w_benign
    maximizes damage while staying close enough to benign distribution to bypass defenses.
    """

    def __init__(self, attack_params=None):
        super().__init__(attack_params)
        # search range for the scalar (usually negative to invert direction)
        self.scaling_min = attack_params.get('scaling_min', -100.0)
        self.scaling_max = attack_params.get('scaling_max', 100.0)
        self.num_steps = attack_params.get('num_steps', 200)

    def attack(self, model, benign_models=None, **kwargs):
        # mb requires knowledge of benign models (or an estimate) to cheat defenses like Krum
        if benign_models is None or len(benign_models) == 0:
            # fallback: simple scaling if no knowledge (degenerate case)
            return self._simple_scale(model, -10.0)

        # 1. compute benign mean (Oracle/Approximated)
        # MB typically assumes attacker knows the benign aggregate to optimize against Krum
        benign_mean = self._aggregate_models(benign_models)

        # 2. search for optimal scalar lambda
        # we want w_poison = lambda * w_local (or lambda * -gradient)
        # goal: maximize distance from global model (damage)
        # constraint: must be selected by Krum/Defense (close to benign models)

        best_scalar = 1.0
        max_damage = -1.0

        # grid search for scalar (as described in paper "iterative method")
        for lam in np.linspace(self.scaling_min, self.scaling_max, self.num_steps):
            if abs(lam) < 1e-4: continue

            # create candidate
            poison_cand = copy.deepcopy(model)
            self._scale_model(poison_cand, lam)

            # check if this candidate bypasses defense ( Krum for example)
            # we approximate this by checking distance to benign mean
            # if distance(poison, mean) <= distance(benign_max, mean), it likely passes
            if self._is_likely_accepted(poison_cand, benign_models, benign_mean):
                # damage metric: usually deviation from true update
                damage = self._compute_damage(poison_cand, benign_mean)
                if damage > max_damage:
                    max_damage = damage
                    best_scalar = lam

        # 3. apply best scalar
        self._scale_model(model, best_scalar)

        return {
            'attack_success': True,
            'scaling_factor': best_scalar
        }

    def _scale_model(self, model, scalar):
        for param in model.parameters():
            param.data *= scalar

    def _aggregate_models(self, models):
        # simple FedAvg of benign models
        avg_model = copy.deepcopy(models[0])
        for p in avg_model.parameters():
            p.data.zero_()

        for m in models:
            for p_avg, p_m in zip(avg_model.parameters(), m.parameters()):
                p_avg.data += p_m.data / len(models)
        return avg_model

    def _is_likely_accepted(self, poison, benign_models, mean):
        # heuristic: os poison closer to mean than the furthest benign model?
        # this mimics Krum's acceptance region roughly
        # (real MB solves strictly for Krum, but this is a standard approx)
        d_poison = self._dist(poison, mean)

        d_benign_max = 0
        for b in benign_models:
            d = self._dist(b, mean)
            if d > d_benign_max:
                d_benign_max = d

        # if we are within the benign "radius", we are likely accepted
        return d_poison <= d_benign_max

    def _dist(self, m1, m2):
        d = 0.0
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            d += torch.norm(p1.data - p2.data, p=2).item()**2
        return d**0.5

    def _compute_damage(self, poison, benign_mean):
        # simple damage: distance from the "true" update (benign mean)
        return self._dist(poison, benign_mean)

    def _simple_scale(self, model, scalar):
        self._scale_model(model, scalar)
        return {'attack_success': True, 'scaling_factor': scalar}
