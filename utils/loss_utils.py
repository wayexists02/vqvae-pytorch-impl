import torch
from torch.nn import functional as F


def compute_vq_loss(x, x_rec, z_e, z_q, EMA=False):
    loss_rec = torch.mean(torch.abs(x - x_rec))
    loss_commitment = torch.mean((z_e - z_q.detach())**2)
    loss_vq = torch.FloatTensor([0]).squeeze().to(x.device)

    if EMA is False:
        loss_vq = torch.mean((z_e.detach() - z_q)**2)

    return loss_rec, loss_commitment, loss_vq
