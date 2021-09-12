import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, weigths=1, dim=-1):
    y = weigths * logits + weigths * sample_gumbel(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=dim)


def gumbel_top2k(logits, temperature, dim=-1):
    logits = torch.nn.functional.softmax(logits, dim=-1)
    g1 = gumbel_softmax_sample(logits, temperature, dim=dim)
    g2 = gumbel_softmax_sample(logits, temperature, weigths=(1 - g1), dim=dim)
    return g1 + g2
