import torch
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, latent_size, latent_res, batch_size):
        super(GaussianNoise, self).__init__()
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.latent_res = latent_res

    def __next__(self):
        return torch.Tensor(torch.Size((self.batch_size, self.latent_size, self.latent_res, self.latent_res))).normal_()
