"""Utility functions for PyTorch.

A collection of common functions that are used by Pytorch algos.

This collection of functions can be used to manage the following:
    - Pytorch GPU usage
        - setting the default Pytorch GPU
        - converting Tensors to GPU Tensors
        - Converting Tensors into `numpy.ndarray` format and vice versa
    - Updating model parameters
"""
import torch

_USE_GPU = False
_DEVICE = None
_GPU_ID = 0


def dict_np_to_torch(array_dict):
    """Convert a dict whose values are numpy arrays to PyTorch tensors.

    Modifies array_dict in place.

    Args:
        array_dict (dict): Dictionary of data in numpy arrays

    Returns:
        dict: Dictionary of data in PyTorch tensors

    """
    for key, value in array_dict.items():
        array_dict[key] = torch.from_numpy(value).float().to(global_device())
    return array_dict


def torch_to_np(tensors):
    """Convert PyTorch tensors to numpy arrays.

    Args:
        tensors (tuple): Tuple of data in PyTorch tensors.

    Returns:
        tuple[numpy.ndarray]: Tuple of data in numpy arrays.

    Note: This method is deprecated and now replaced by
        `garage.torch.utils.to_numpy`.

    """
    value_out = tuple(v.numpy() for v in tensors)
    return value_out


def flatten_batch(tensor):
    """Flatten a batch of observations.

    Reshape a tensor of size (X, Y, Z) into (X*Y, Z)

    Args:
        tensor (torch.Tensor): Tensor to flatten.

    Returns:
        torch.Tensor: Flattened tensor.

    """
    return tensor.reshape((-1, ) + tensor.shape[2:])


def update_module_params(module, new_params):  # noqa: D202
    """Load parameters to a module.

    This function acts like `torch.nn.Module._load_from_state_dict()`, but
    it replaces the tensors in module with those in new_params, while
    `_load_from_state_dict()` loads only the value. Use this function so
    that the `grad` and `grad_fn` of `new_params` can be restored

    Args:
        module (torch.nn.Module): A torch module.
        new_params (dict): A dict of torch tensor used as the new
            parameters of this module. This parameters dict should be
            generated by `torch.nn.Module.named_parameters()`

    """

    # pylint: disable=protected-access
    def update(m, name, param):
        del m._parameters[name]  # noqa: E501
        setattr(m, name, param)
        m._parameters[name] = param  # noqa: E501

    named_modules = dict(module.named_modules())

    for name, new_param in new_params.items():
        if '.' in name:
            module_name, param_name = tuple(name.rsplit('.', 1))
            if module_name in named_modules:
                update(named_modules[module_name], param_name, new_param)
        else:
            update(module, name, new_param)


def set_gpu_mode(mode, gpu_id=0):
    """Set GPU mode and device ID.

    Args:
        mode (bool): Whether or not to use GPU
        gpu_id (int): GPU ID

    """
    # pylint: disable=global-statement
    global _GPU_ID
    global _USE_GPU
    global _DEVICE
    _GPU_ID = gpu_id
    _USE_GPU = mode
    _DEVICE = torch.device(('cuda:' + str(_GPU_ID)) if _USE_GPU else 'cpu')


def global_device():
    """Returns the global device that torch.Tensors should be placed on.

    Note: The global device is set by using the function
        `garage.torch.utils.set_gpu_mode.` If this functions is never called
        `garage.torch.utils.device()` returns None.

    Returns:
        `torch.Device`: The global device that newly created torch.Tensors
            should be placed on.

    """
    # pylint: disable=global-statement
    global _DEVICE
    return _DEVICE
