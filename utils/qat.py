from __future__ import print_function
import torch

# quantifu every layers by the different ratio
def qat_weight_inlayers(state_dict, bits, act_thresh):
    max_value = 2**(bits-1) - 1

    qat_thresh = []
    for layer, tensor in state_dict.items():
        max_abs = act_thresh + 0.1
        tmp = torch.max(torch.abs(tensor)).item()
        max_abs = tmp if tmp > max_abs else max_abs
        ratio = max_value / max_abs
        state_dict[layer] = torch.round(state_dict[layer] * ratio)
        qat_thresh.append(round(ratio * act_thresh))
    return state_dict, qat_thresh


# quantifu all layers by the same ratio
def qat_weight(state_dict, bits, act_thresh):
    max_value = 2**(bits-1) - 1
    qat_thresh = []
    # find max abs
    max_abs = act_thresh + 0.1
    for _, tensor in state_dict.items():
        tmp = torch.max(torch.abs(tensor)).item()
        max_abs = tmp if tmp > max_abs else max_abs
    ratio = max_value / max_abs
    for layer, tensor in state_dict.items():
        state_dict[layer] = torch.round(state_dict[layer] * ratio)
        qat_thresh.append(round(ratio * act_thresh))
    return state_dict, qat_thresh

