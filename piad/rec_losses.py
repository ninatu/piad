import os
from collections import defaultdict, OrderedDict
import torch
from torch import nn

from piad.feature_extractor import PretrainedVGG19FeatureExtractor


class PerceptualLoss(torch.nn.Module):
    def __init__(self,
                 reduction='mean',
                 img_weight=0,
                 feature_weights=None,
                 use_feature_normalization=False,
                 use_L1_norm=False,
                 use_relative_error=False):
        super(PerceptualLoss, self).__init__()
        """
        We assume that input is normalized with 0.5 mean and 0.5 std
        """

        assert reduction in ['none', 'sum', 'mean']

        MEAN_VAR_ROOT = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'vgg19_ILSVRC2012_object_detection_mean_var.pt')

        self.vgg19_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.vgg19_std = torch.Tensor([0.229, 0.224, 0.225])

        if use_feature_normalization:
            self.mean_var_dict = torch.load(MEAN_VAR_ROOT)
        else:
            self.mean_var_dict = defaultdict(
                lambda: (torch.tensor([0.0], requires_grad=False), torch.tensor([1.0], requires_grad=False))
            )

        self.reduction = reduction
        self.use_L1_norm = use_L1_norm
        self.use_relative_error = use_relative_error

        self.img_weight = img_weight
        if feature_weights is None:
            self.feature_weights = OrderedDict({})
        else:
            self.feature_weights = OrderedDict(feature_weights)

        self.model = PretrainedVGG19FeatureExtractor()

    def set_reduction(self, reduction):
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction

    def forward(self, x, y):
        layers = list(self.feature_weights.keys())
        weights = list(self.feature_weights.values())

        x = self._preprocess(x)
        y = self._preprocess(y)

        f_x = self.model(x, layers)
        f_y = self.model(y, layers)

        loss = None

        if self.img_weight != 0:
            loss = self.img_weight * self._loss(x, y)

        for i in range(len(f_x)):
            # put mean, var on right device
            mean, var = self.mean_var_dict[layers[i]]
            mean, var = mean.to(f_x[i].device), var.to(f_x[i].device)
            self.mean_var_dict[layers[i]] = (mean, var)

            # compute loss
            norm_f_x_val = (f_x[i] - mean) / var
            norm_f_y_val = (f_y[i] - mean) / var

            cur_loss = self._loss(norm_f_x_val, norm_f_y_val)

            if loss is None:
                loss = weights[i] * cur_loss
            else:
                loss += weights[i] * cur_loss

        loss /= (self.img_weight + sum(weights))

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise NotImplementedError('Not implemented reduction: {:s}'.format(self.reduction))

    def _preprocess(self, x):
        assert len(x.shape) == 4

        if x.shape[1] != 3:
            x = x.expand(-1, 3, -1, -1)

        # denormalize
        vector = torch.Tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1).to(x.device)
        x = x * vector + vector

        # normalize
        x = (x - self.vgg19_mean.reshape(1, 3, 1, 1).to(x.device)) / self.vgg19_std.reshape(1, 3, 1, 1).to(x.device)
        return x

    def _loss(self, x, y):
        if self.use_L1_norm:
            norm = lambda z: torch.abs(z)
        else:
            norm = lambda z: z * z

        diff = (x - y)

        if not self.use_relative_error:
            loss = norm(diff)
        else:
            means = norm(x).mean(3).mean(2).mean(1)
            means = means.detach()
            loss = norm(diff) / means.reshape((means.size(0), 1, 1, 1))

        return loss.mean(3).mean(2).mean(1)


class L2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L2Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def set_reduction(self, reduction):
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def forward(self, x, y):
        assert len(x.shape) == 4
        losses = ((x - y) * (x - y)).sum(3).sum(2).sum(1) / (x.size(1) * x.size(2) * x.size(3))
        if self._reduction == 'none':
            return losses
        elif self._reduction == 'mean':
            return torch.mean(losses)
        else:
            return torch.sum(losses)


class L1Loss(nn.Module):
    def __init__(self, reduction='none'):
        super(L1Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def set_reduction(self, reduction):
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def forward(self, x, y):
        assert len(x.shape) == 4
        losses = (torch.abs(x - y)).sum(3).sum(2).sum(1) / (x.size(1) * x.size(2) * x.size(3))
        if self._reduction == 'none':
            return losses
        elif self._reduction == 'mean':
            return torch.mean(losses)
        else:
            return torch.sum(losses)
