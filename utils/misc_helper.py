import torch


def to_device(input_dict, device=None):
    """Move input dict tensors to device; keep non-tensors unchanged.
    Expected keys include 'image', 'gt_image', 'mask', but this is generic.
    """
    if device is None:
        # Try infer device from any tensor value
        for v in input_dict.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
    out = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out 