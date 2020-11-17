import torch
from torch import nn


def get_norm_layer(layer_type, **kwargs):
    if layer_type == 'none':
        return []
    elif layer_type == 'bn':
        return [nn.BatchNorm2d(kwargs['num_features'])]
    elif layer_type == 'in':
        return [nn.InstanceNorm2d(kwargs['num_features'])]
    else:
        raise NotImplementedError("Unknown type: {}".format(layer_type))


def get_act_layer(layer_type, **kwargs):
    if layer_type == 'relu':
        return [nn.ReLU()]
    elif layer_type == 'leaky_relu':
        return [nn.LeakyReLU(kwargs.get('negative_slope', 0.2), inplace=False)]
    elif layer_type == 'tanh':
        return [nn.Tanh()]
    elif layer_type == 'sigmoid':
        return [nn.Sigmoid()]
    elif layer_type == 'linear':
        return []
    else:
        raise NotImplementedError


def get_pool_layer(type, **kwargs):
    if type == 'avg':
        return [nn.AvgPool2d(kwargs.get('kernel_size', 2), kwargs.get('stride', 2))]
    elif type == 'max':
        return [nn.MaxPool2d(kwargs.get('kernel_size', 2), kwargs.get('stride', 2))]
    else:
        raise NotImplementedError


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm='none', act='linear'):
        super(ConvBlock, self).__init__()
        leaky_relu_param = 0.2
        layers = []

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain(act, param=leaky_relu_param))
        layers.append(conv)

        layers += get_norm_layer(norm, num_features=out_channels)
        layers += get_act_layer(act, negative_slope=leaky_relu_param)

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu'):
        super(ResBlock, self).__init__()

        model = []
        model += [ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act)]
        model += [ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear')]
        self.model = nn.Sequential(*model)

        if input_dim == output_dim:
            self.skipcon = nn.Sequential()
        else:
            self.skipcon = ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear')

    def forward(self, x):
        return self.skipcon(x) + self.model(x)


class PreActResnetBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu'):
        super(PreActResnetBlock, self).__init__()

        model = []
        model += get_norm_layer(norm, num_features=input_dim)
        model += get_act_layer(act)
        model += [
            ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act),
            ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear')
        ]
        self.model = nn.Sequential(*model)

        if input_dim == output_dim:
            self.skipcon = nn.Sequential()
        else:
            self.skipcon = ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear')

    def forward(self, x):
        return self.skipcon(x) + self.model(x)


class PreActResnetBlockUp(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu'):
        super(PreActResnetBlockUp, self).__init__()

        model = []
        model += get_norm_layer(norm, num_features=input_dim)
        model += get_act_layer(act)
        model += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act),
            ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear')
        ]
        self.model = nn.Sequential(*model)

        skipcon = [nn.Upsample(scale_factor=2, mode='nearest')]
        if input_dim != output_dim:
            skipcon += [ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear')]
        self.skipcon = nn.Sequential(*skipcon)

    def forward(self, x):
        return self.skipcon(x) + self.model(x)


class PreActResnetBlockDown(nn.Module):
    def __init__(self, input_dim, output_dim, norm='bn', act='relu', pool='avg'):
        super(PreActResnetBlockDown, self).__init__()

        model = []
        model += get_norm_layer(norm, num_features=input_dim)
        model += get_act_layer(act)
        model += [
            ConvBlock(input_dim, output_dim, 3, 1, 1, norm=norm, act=act),
            ConvBlock(output_dim, output_dim, 3, 1, 1, norm='none', act='linear'),
        ]
        model += get_pool_layer(pool, kernel_size=2, stride=2)
        self.model = nn.Sequential(*model)

        skipcon = get_pool_layer(pool, kernel_size=2, stride=2)
        if input_dim != output_dim:
            skipcon += [ConvBlock(input_dim, output_dim, 1, 1, 0, norm='none', act='linear')]
        self.skipcon = nn.Sequential(*skipcon)

    def forward(self, x):
        return self.skipcon(x) + self.model(x)


class EqualLayer(nn.Module):
    def forward(self, x):
        return x


class MinibatchStdDevLayer(nn.Module):
    def __init__(self, save_vals=False):
        super(MinibatchStdDevLayer, self).__init__()
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)
        self.save_vals = save_vals
        self.saved_vals = None

    def forward(self, x):
        target_shape = list(x.size())
        target_shape[1] = 1

        vals = self.adjusted_std(x, dim=0, keepdim=True)
        vals = torch.mean(vals)

        if self.save_vals:
            self.saved_vals = vals.data

        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)
