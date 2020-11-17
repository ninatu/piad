from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import autograd as autograd


class GANLoss(ABC):
    @abstractmethod
    def gen_loss(self, dis, real_x=None, fake_x=None):
        """
        :return: (loss, loss_info), where
            loss is Tensor,
            loss_info is a dict {'subloss_name': subloss_value, ...} for logging. By default empty dict
        """
        pass

    @abstractmethod
    def dis_loss(self, dis, real_x=None, fake_x=None):
        """
        :return: (loss, loss_info), where
            loss is Tensor,
            loss_info is a dict {'subloss_name': subloss_value, ...} for logging. By default empty dict
        """
        pass

    @abstractmethod
    def dis_gen_loss(self, dis, real_x=None, fake_x=None):
        """
        :return: ((gen_loss, gen_loss_info), (dis_loss, dis_loss_info))
        """
        pass


class WassersteinLoss(GANLoss):
    def __init__(self, gradient_penalty, norm_penalty, lambd=1.0, wass_target_gamma=1.0):
        self._wass_gp = gradient_penalty
        self._wass_epsilon = norm_penalty
        self._wass_lambda = lambd
        self._wass_target_gamma = wass_target_gamma

    def gen_loss(self, dis, real_x=None, fake_x=None):
        assert fake_x is not None
        loss = _wasserstein_loss(dis, x=real_x, tilda_x=fake_x)
        return loss, {'loss': loss.data.item()}

    def dis_loss(self, dis, real_x=None, fake_x=None):
        wass_loss, norm = _wasserstein_loss(dis, x=real_x, tilda_x=fake_x, return_norm=True)
        norm = self._wass_epsilon * norm
        wass_loss = - self._wass_lambda * wass_loss

        with torch.set_grad_enabled(True):
            gp_loss = self._wass_gp * _gradient_penalty(dis, real_x, fake_x, self._wass_target_gamma)
        total = wass_loss + norm + gp_loss

        loss_info = {
            'wasserstein_loss': wass_loss.data.item(),
            'norm_penalty': norm.data.item(),
            'gradient_penalty': gp_loss.data.item(),
        }

        return total, loss_info

    def dis_gen_loss(self, dis, real_x=None, fake_x=None):
        wass_loss, norm = _wasserstein_loss(dis, x=real_x, tilda_x=fake_x, return_norm=True)

        gen_total = wass_loss
        gen_loss_info = {'loss': gen_total.data.item()}

        norm = self._wass_epsilon * norm
        dis_wass_loss = - self._wass_lambda * wass_loss

        with torch.set_grad_enabled(True):
            gp_loss = self._wass_gp * _gradient_penalty(dis, real_x, fake_x, self._wass_target_gamma)

        dis_total = dis_wass_loss + norm + gp_loss
        dis_loss_info = {
            'wasserstein_loss': dis_wass_loss.data.item(),
            'norm_penalty': norm.data.item(),
            'gradient_penalty': gp_loss.data.item(),
        }

        return (dis_total, dis_loss_info), (gen_total, gen_loss_info)


class LsLoss(GANLoss):
    def __init__(self, use_mulnoise=False):
        self._use_mulnoise = use_mulnoise
        self._mean_exp_output = None

    def gen_loss(self, dis, real_x=None, fake_x=None):
        loss = _mse_loss(dis, x=fake_x)
        return loss, {'loss': loss.data.item()}

    def dis_loss(self, dis, real_x=None, fake_x=None):
        loss, cur_mean_output = _mse_loss(dis, real_x, fake_x, return_mean_output=True)

        if self._use_mulnoise:
            if self._mean_exp_output is None:
                strength = dis.strength
                self._mean_exp_output = (strength / 0.2) + 0.5

            self._mean_exp_output = 0.1 * cur_mean_output + 0.9 * self._mean_exp_output
            strength = 0.2 * max(0, self._mean_exp_output - 0.5)
            dis.set_strength(strength)

        return loss, {'loss': loss.data.item()}

    def dis_gen_loss(self, dis, real_x=None, fake_x=None):
        raise NotImplementedError()


def _wasserstein_loss(dis, x=None, tilda_x=None, return_norm=False):
    loss = 0
    norm = 0
    if x is not None:
        dis_x = dis(x)
        loss += dis_x.mean()
        if return_norm:
            norm += (dis_x ** 2).mean()
        del dis_x
    if tilda_x is not None:
        dis_tilda_x = dis(tilda_x)
        loss -= dis_tilda_x.mean()
        if return_norm:
            norm += (dis_tilda_x ** 2).mean()
        del dis_tilda_x
    if return_norm:
        return loss, norm
    else:
        return loss


def _mse_loss(dis, x=None, tilda_x=None, return_mean_output=False, smooth_labels=False):
    mse = torch.nn.MSELoss()
    loss = []
    output = []
    if x is not None:
        pred_label = dis(x)
        if smooth_labels:
            real_label = autograd.Variable(torch.Tensor(pred_label.size()).uniform_(0.8, 1.2).cuda())
        else:
            real_label = autograd.Variable(torch.Tensor(pred_label.size()).fill_(1).cuda())
        loss.append(mse(pred_label, real_label))
        output.extend(pred_label.data.cpu().numpy().tolist())
        del real_label, pred_label
    if tilda_x is not None:
        pred_label = dis(tilda_x)
        if smooth_labels:
            fake_label = autograd.Variable(torch.Tensor(pred_label.size()).uniform_(-0.2, 0.2).cuda())
        else:
            fake_label = autograd.Variable(torch.Tensor(pred_label.size()).fill_(0).cuda())
        loss.append(mse(pred_label, fake_label))
        output.extend(pred_label.data.cpu().numpy().tolist())
        del fake_label, pred_label
    if return_mean_output:
        mean_output = np.array(output).mean()
        return sum(loss) / len(loss), mean_output
    else:
        return sum(loss) / len(loss)


def _gradient_penalty(dis, x, tilda_x, target_gamma=1.0):
    batch_size = x.size(0)
    alpha = torch.rand(batch_size, 1).view(batch_size, 1, 1, 1).cuda()
    hat_x = alpha * x.data + ((1 - alpha) * tilda_x.data)
    hat_x = hat_x.detach().requires_grad_(True)
    hat_y = dis(hat_x)

    gradients = autograd.grad(outputs=hat_y,
                              inputs=hat_x,
                              grad_outputs=torch.ones(hat_y.size()).cuda(),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty_value = (((gradients.norm(2, dim=1) - target_gamma) ** 2) / (target_gamma ** 2)).mean()
    return gradient_penalty_value
