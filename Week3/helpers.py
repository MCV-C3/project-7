import torch


def detect_mnasnet_type(block):
    """
    detects type of mnasnet block
    pytorch implementation -> green == blue
    one of {"green", "yellow", "brown"}
    """
    layers = block.layers

    expand = layers[0]
    dw = layers[3]

    in_ch = expand.in_channels
    mid_ch = expand.out_channels
    expansion = mid_ch // in_ch

    k = dw.kernel_size[0]

    if expansion == 6 and k == 3:
        return "green"
    elif expansion == 6 and k == 5:
        return "yellow"
    elif expansion == 3 and k == 5:
        return "brown"
    else:
        raise ValueError("Unknown MNASNet block type")


def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True

