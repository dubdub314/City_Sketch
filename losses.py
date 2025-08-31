import torch
import torch.nn.functional as F

def nt_xent(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def supcon_loss(feats, labels, temperature=0.2):
    feats = F.normalize(feats, dim=-1)
    sim = feats @ feats.t() / temperature
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    logits = sim - torch.max(sim, dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits) * (~torch.eye(feats.size(0), dtype=torch.bool, device=feats.device))
    log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True)+1e-9)
    mask_pos = mask & (~torch.eye(feats.size(0), dtype=torch.bool, device=feats.device))
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1)+1e-9)
    loss = -mean_log_prob_pos.mean()
    return loss

def pair_loss(logit, target, kind='bce'):
    if kind=='bce':
        return F.binary_cross_entropy_with_logits(logit, target)
    elif kind=='mse':
        return F.mse_loss(torch.sigmoid(logit), target)
    elif kind=='margin':
        margin = 0.2
        pos = (target>0.5).float()
        neg = 1-pos
        s = torch.sigmoid(logit)
        return (neg*F.relu(s - (1-margin)) + pos*F.relu((margin) - s)).mean()
    else:
        raise ValueError(f'Unknown pair loss: {kind}')

def node_align_loss(Ha, Hb, pairs, reduction='mean'):
    if not pairs: return Ha.new_tensor(0.0)
    sa = F.normalize(Ha, dim=-1); sb = F.normalize(Hb, dim=-1)
    vals = [(1 - (sa[ia]*sb[ib]).sum(dim=-1)) for (ia,ib) in pairs]
    v = torch.stack(vals)
    return v.mean() if reduction=='mean' else v.sum()
