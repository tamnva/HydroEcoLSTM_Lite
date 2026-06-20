
import torch
import torch.nn as nn


class revin(nn.Module):
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, n_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, n_features))

    def forward(self, x, mode):
        # x: (batch, seq_len, features)

        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = x.std(dim=1, keepdim=True, unbiased=False).detach()
            x = (x - self.mean) / (self.std + self.eps)

            if self.affine:
                x = x * self.gamma + self.beta

            return x

        elif mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)

            x = x * (self.std + self.eps) + self.mean
            return x