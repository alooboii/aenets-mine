import torch
import random
import numpy as np

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_module(model, module_name):
    """
    Get a module from a model by its name (dot-separated path).
    Handles both attribute access and index access (e.g., 'layer1[0].conv1').
    
    Args:
        model (nn.Module): The model to extract the module from
        module_name (str): Path to the module (e.g., 'layer1.0.conv1' or 'layer1[0].conv1')
        
    Returns:
        nn.Module: The requested module
    """
    # Normalize the path: convert brackets to dots
    normalized_name = module_name.replace('[', '.').replace(']', '')
    modules = normalized_name.split('.')
    
    current_module = model
    
    for module in modules:
        if not module:  # Skip empty strings from double dots
            continue
            
        # Handle integer indices for sequential containers
        if module.isdigit():
            current_module = current_module[int(module)]
        else:
            current_module = getattr(current_module, module)
    
    return current_module


def get_weight_shape(model, layer_path):
    """
    Get weight shape of a specific layer using precise path specification.
    
    Args:
        model: The model (e.g., ResNet)
        layer_path: Precise path like "layer3[0].conv1" or "layer2[1].conv2"
    
    Returns:
        tuple: Shape of the weight tensor
    """
    module = get_module(model, layer_path)
    
    if not hasattr(module, 'weight'):
        raise AttributeError(f"Module at path '{layer_path}' has no 'weight' attribute")
    
    return tuple(module.weight.shape)


def count_params(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): The model to count parameters for
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


