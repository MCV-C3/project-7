import torch
import torch.nn as nn


def add_dropout_after_block(module: nn.Module, p: float) -> nn.Module:
    return nn.Sequential(
        module,
        nn.Dropout2d(p=p)
    )

def add_dropout_after_conv(module: nn.Module, p: float) -> nn.Module:
    # Case 1: bare Conv2d
    if isinstance(module, nn.Conv2d):
        return nn.Sequential(
            module,
            nn.Dropout2d(p=p)
        )
    # Case 2: Sequential
    if isinstance(module, nn.Sequential):
        new_layers = []
        for layer in module:
            new_layer = add_dropout_after_conv(layer, p)
            new_layers.append(new_layer)
            # If layer itself was Conv2d (already wrapped above, so skip)
            if isinstance(layer, nn.Conv2d):
                continue
        return nn.Sequential(*new_layers)
    # Case 3: InvertedResidual
    if hasattr(module, "layers") and isinstance(module.layers, nn.Sequential):
        module.layers = add_dropout_after_conv(module.layers, p)
    return module



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

def add_batchnorm_after_block(block: nn.Module) -> nn.Module:
    """
    Adds a BatchNorm2d layer AFTER a convolutional block.
    Assumes the block outputs a 4D tensor (N, C, H, W).
    """
    # Infer output channels by running a dummy tensor (safe)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        try:
            out = block(dummy)
        except:
            return block  # safety fallback

    out_channels = out.shape[1]

    return nn.Sequential(
        block,
        nn.BatchNorm2d(out_channels)
    )
