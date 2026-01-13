import torch
import copy
from collections import OrderedDict


def flatten_model(model):
    """Flatten model parameters into a 1D tensor."""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)


def unflatten_model(flattened_params, model):
    """Unflatten 1D tensor back into model parameters."""
    pointer = 0
    for param in model.parameters():
        num_params = param.numel()
        param.data = flattened_params[pointer:pointer + num_params].view(param.shape)
        pointer += num_params


def model_to_dict(model):
    """Convert model parameters to a dictionary."""
    return OrderedDict(model.named_parameters())


def dict_to_model(param_dict, model):
    """Load parameters from dictionary into model."""
    model.load_state_dict(param_dict, strict=False)


def get_model_norm(model, norm_type=2):
    """Calculate L2 or L1 norm of model parameters."""
    total_norm = 0
    for param in model.parameters():
        if norm_type == 2:
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
        elif norm_type == 1:
            total_norm += param.data.norm(1).item()
    
    if norm_type == 2:
        return total_norm ** 0.5
    else:
        return total_norm


def cosine_similarity(model1, model2):
    """Calculate cosine similarity between two models."""
    flat1 = flatten_model(model1)
    flat2 = flatten_model(model2)
    
    dot_product = torch.dot(flat1, flat2)
    norm1 = torch.norm(flat1)
    norm2 = torch.norm(flat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return (dot_product / (norm1 * norm2)).item()


def euclidean_distance(model1, model2):
    """Calculate Euclidean distance between two models."""
    flat1 = flatten_model(model1)
    flat2 = flatten_model(model2)
    return torch.norm(flat1 - flat2).item()


def model_average(models, weights=None):
    """Average multiple models with optional weights."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    averaged_state_dict = OrderedDict()
    
    for name, param in models[0].named_parameters():
        averaged_param = torch.zeros_like(param)
        for i, model in enumerate(models):
            model_param = dict(model.named_parameters())[name]
            averaged_param += weights[i] * model_param
        averaged_state_dict[name] = averaged_param
    
    return averaged_state_dict


def clip_model_norm(model, max_norm):
    """Clip model parameters to have maximum norm."""
    current_norm = get_model_norm(model, norm_type=2)
    if current_norm > max_norm:
        scaling_factor = max_norm / current_norm
        for param in model.parameters():
            param.data *= scaling_factor
    return current_norm


def group_parameters_by_layer(model):
    """Group model parameters by layer type."""
    conv_params = []
    bn_params = []
    fc_params = []
    
    for name, param in model.named_parameters():
        if 'conv' in name.lower():
            conv_params.append(param)
        elif 'bn' in name.lower() or 'batch' in name.lower():
            bn_params.append(param)
        elif 'fc' in name.lower() or 'linear' in name.lower():
            fc_params.append(param)
    
    return {
        'conv': conv_params,
        'bn': bn_params,
        'fc': fc_params
    }