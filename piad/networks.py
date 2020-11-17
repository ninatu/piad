import math
from torch import nn

from piad.layers import MinibatchStdDevLayer, ConvBlock,\
    PreActResnetBlockUp, PreActResnetBlockDown, get_act_layer,\
    ResBlock, PreActResnetBlock


def is_power2(num):
    return ((num & (num - 1)) == 0) and num != 0


class ResNetGenerator(nn.Module):
    def __init__(self, image_res, image_dim, latent_res, latent_dim, inner_dims):
        super(ResNetGenerator, self).__init__()

        if not(is_power2(latent_res) and is_power2(image_res)):
            raise ValueError("Generator. latent_res and image_res must be power of 2. "
                             f"Given values: latent_res={latent_res}, image_res={image_res}")

        len_inner_dims = math.log2(image_res * 2 / max(latent_res, 4))
        if len(inner_dims) != len_inner_dims:
            raise ValueError(f"Generator. for latent_res={latent_res} and image_res={image_res} "
                             f"you should specify inner_dims with length={len_inner_dims}. "
                             f"Given value: inner_dims={inner_dims} with length={len(inner_dims)}")

        self.image_res = image_res
        self.image_dim = image_dim
        self.latent_res = latent_res
        self.latent_dim = latent_dim

        layers = []
        n_layer = 0

        # add the first layer
        if self.latent_res == 1:
            layers += [ConvBlock(self.latent_dim, inner_dims[n_layer], kernel_size=4, stride=1, padding=3,
                                 norm='none', act='linear')]
            cur_resolution = 4
        else:
            layers += [ResBlock(self.latent_dim, inner_dims[n_layer], norm='none', act='leaky_relu')]
            cur_resolution = self.latent_res

        # add the intermediate layers
        while cur_resolution < self.image_res:
            layers += [PreActResnetBlockUp(input_dim=inner_dims[n_layer],
                                          output_dim=inner_dims[n_layer + 1],
                                          norm='none',
                                          act='leaky_relu')]
            n_layer += 1
            cur_resolution *= 2

        # postprocessing
        layers += [PreActResnetBlock(inner_dims[n_layer], inner_dims[n_layer], norm='none', act='leaky_relu')]
        layers += get_act_layer('leaky_relu')
        layers += [ConvBlock(inner_dims[n_layer], self.image_dim, kernel_size=1, stride=1, padding=0,
                             norm='none',
                             act='linear')]
        self.model = self._init_layers(nn.Sequential(*layers))

    def _init_layers(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu', a=0.2)
        return model

    def forward(self, x):
        return self.model(x)


class ResNetEncoder(nn.Module):
    def __init__(self, image_res, image_dim, latent_res, latent_dim, inner_dims, use_mbstddev=False):
        super(ResNetEncoder, self).__init__()

        if not (is_power2(latent_res) and is_power2(image_res)):
            raise ValueError("Encoder. latent_res and image_res must be power of 2. "
                             f"Given values: latent_res={latent_res}, image_res={image_res}")

        len_inner_dims = math.log2(image_res * 2 / max(latent_res, 4))
        if len(inner_dims) != len_inner_dims:
            raise ValueError(f"Encoder. for latent_res={latent_res} and image_res={image_res} "
                             f"you should specify inner_dims with length={len_inner_dims}. "
                             f"Given value: inner_dims={inner_dims} with length={len(inner_dims)}")

        self.image_res = image_res
        self.image_dim = image_dim
        self.latent_res = latent_res
        self.latent_dim = latent_dim
        layers = []
        n_layer = 0

        # pre-processing
        layers += [ConvBlock(in_channels=self.image_dim,
                             out_channels=inner_dims[n_layer],
                             kernel_size=3, stride=1, padding=1, norm='none', act='linear')]
        cur_resolution = self.image_res

        # add the intermediate layers
        inner_dims = [inner_dims[0]] + inner_dims
        while cur_resolution > max(4, self.latent_res):
            layers += [PreActResnetBlockDown(input_dim=inner_dims[n_layer],
                                             output_dim=inner_dims[n_layer + 1],
                                             norm='none',
                                             act='leaky_relu',
                                             pool='avg')]
            n_layer += 1
            cur_resolution /= 2

        # add the last layer
        if use_mbstddev:
            layers += [MinibatchStdDevLayer()]

        prev_dim = inner_dims[n_layer] + use_mbstddev

        if self.latent_res == 1:
            layers += get_act_layer('leaky_relu')
            layers += [ConvBlock(prev_dim, inner_dims[n_layer + 1],
                                 kernel_size=4, stride=1, padding=0, norm='none', act='leaky_relu')]
        else:
            layers += [PreActResnetBlock(prev_dim, inner_dims[n_layer + 1], norm='none', act='leaky_relu')]
            layers += get_act_layer('leaky_relu')

        layers += [ConvBlock(inner_dims[n_layer + 1], self.latent_dim, kernel_size=1, stride=1, padding=0,
                             norm='none', act='linear')]

        self.model = self._init_layers(nn.Sequential(*layers))

    def _init_layers(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu', a=0.2)
        return model

    def forward(self, x):
        return self.model(x)


class LatentDiscriminator(nn.Module):
    def __init__(self, input_dim, inner_dims):
        super(LatentDiscriminator, self).__init__()

        layers = list()
        inner_dims = [input_dim] + inner_dims
        for n_layer in range(len(inner_dims) - 1):
                layers += [ConvBlock(inner_dims[n_layer], inner_dims[n_layer + 1],
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 norm='none',
                                 act='leaky_relu')
            ]

        layers.append(ConvBlock(inner_dims[-1], 1,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm='none',
                                act='linear'))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten vector
        x = x.reshape(x.size(0), -1, 1, 1)
        return self.model(x)
