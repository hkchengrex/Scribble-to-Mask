import torch
import torch.nn.functional as F

def aggregate_wbg_channel(prob, keep_bg=False):
    new_prob = torch.cat([
        torch.prod(1-prob, dim=1, keepdim=True),
        prob
    ], 1).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))

    if keep_bg:
        return logits, F.softmax(logits, dim=1)
    else:
        return logits, F.softmax(logits, dim=1)[:, 1:]
