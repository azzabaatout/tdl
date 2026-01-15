import torch
import copy
import numpy as np
from .base_attack import ModelPoisoningAttack

class LocalAttack(ModelPoisoningAttack):
    """
    Local Attack (LA)

    As described in paper Section 2.1:
    1. Generate random vector of {-1, 1}.
    2. Use iterative method to find shared scalar lambda.
    """

    def __init__(self, attack_params=None):
        super().__init__(attack_params)
        self.max_iterations = attack_params.get('max_iterations', 100)
        # la typically tries to be aggressive.
        self.target_norm_ratio = attack_params.get('target_norm_ratio', 10.0)

    def attack(self, model, **kwargs):
        original_model = copy.deepcopy(model)

        # 1. we generate Perturbation Vector P (consisting of 1 and -1)
        # Note: The perturbation is usually additive to the update
        # but for model poisoning, we perturb the model weights directly.
        perturbation = {}
        for name, param in model.named_parameters():
            # random vector consisting of 1 and -1
            perturbation[name] = torch.sign(torch.randn_like(param.data))

        # 2. we find optimal scalar lambda
        # Goal: Maximize magnitude (norm) or distance while staying "plausible"
        # since la is a benchmark often checked against krum/normclip,
        # we usually just maximize the scalar until we hit the defense bound.
        # if no specific defense bound is known (naive LA), we just pick a large scalar.

        best_scalar = 1.0
        # searcg range: usually 0 to large number
        scalars = np.linspace(0, self.target_norm_ratio, self.max_iterations)

        for scalar in scalars:
            # candidate: w_poison = w_clean - lambda * perturbation
            # a note: LA often subtracts to oppose direction, or adds
            # the random sign vector makes add/sub equivalent on average

            # check if this scalar produces a model that "breaks" the simulation
            # (e.g. NaNs) or is simply the largest one we are testing.

            # in a real benchmark against Krum, we would check krum_score(candidate).
            # here, we assume "naive" LA just picks a large scalar.
            best_scalar = scalar

        # 3. apply attack
        # w_new = w + lambda * P * std_dev(w)
        # (scaling by param magnitude is common to keep perturbations relative)
        for name, param in model.named_parameters():
            # scale perturbation by layer-wise norm to be scale-invariant
            layer_norm = torch.norm(param.data)
            num_params = param.numel()
            # heuristic: Perturb by lambda * avg_magnitude per param
            scale = (layer_norm / (num_params ** 0.5))

            param.data = param.data + (best_scalar * perturbation[name] * scale)

        return {
            'attack_success': True,
            'scaling_factor': best_scalar
        }
