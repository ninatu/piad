from collections import defaultdict
import re
import numpy as np
from torch import nn
from torch.optim import Adam

from piad.utils import ewma_vectorized


class Optimizer(nn.Module):
    def __init__(self, gen_params, enc_params, image_dis_params, latent_dis_params,
                 image_adv_loss, latent_adv_loss, rec_loss, adam_kwargs,
                 enc_rec_loss_weight=None, gen_rec_loss_weight=None, use_grad_norm_policy=True):

        super(Optimizer, self).__init__()

        self.image_adv_loss = image_adv_loss
        self.latent_adv_loss = latent_adv_loss
        self.rec_loss = rec_loss

        assert use_grad_norm_policy == (enc_rec_loss_weight is None and gen_rec_loss_weight is None)
        assert (enc_rec_loss_weight is None) == (gen_rec_loss_weight is None)
        if use_grad_norm_policy:
            self.enc_rec_loss_weight = 1
            self.gen_rec_loss_weight = 1
        else:
            self.enc_rec_loss_weight = enc_rec_loss_weight
            self.gen_rec_loss_weight = gen_rec_loss_weight

        self.use_grad_norm_policy = use_grad_norm_policy
        self.grad_norm_policy_helper = GradientNormalizingWeightPolicyHelper()

        def preprocess(params):
            return [p for p in params if p.requires_grad]

        self.image_dis_opt = Adam(preprocess(image_dis_params), **adam_kwargs['image_dis'])
        self.latent_dis_opt = Adam(preprocess(latent_dis_params), **adam_kwargs['latent_dis'])
        self.enc_gen_opt = Adam(preprocess(enc_params) + preprocess(gen_params), **adam_kwargs['enc_gen'])

    @staticmethod
    def compute_dis_loss(dis, loss, opt, real, fake, update_parameters):
        loss, loss_info = loss.dis_loss(dis, real.detach(), fake.detach())

        if update_parameters:
            opt.zero_grad()
            loss.backward()
            opt.step()

        loss_info['total'] = loss.data.item()
        return loss_info

    @staticmethod
    def compute_adv_loss(dis, loss, real, fake):
        loss, loss_info = loss.gen_loss(dis, real, fake)
        return loss

    def compute_image_dis_loss(self, image_dis, real, fake, update_parameters=True):
        return self.compute_dis_loss(image_dis, self.image_adv_loss, self.image_dis_opt, real, fake,
                                     update_parameters=update_parameters)

    def compute_latent_dis_loss(self, latent_dis, real, fake, update_parameters=True):
        return self.compute_dis_loss(latent_dis, self.latent_adv_loss, self.latent_dis_opt, real, fake,
                                     update_parameters=update_parameters)

    def compute_enc_gen_loss(self, enc, gen, latent_dis, image_dis, fake_x, real_x, fake_z, rec_x,
                             update_parameters=True, update_grads=False, logger=None, n_iter=None):

        image_adv_loss = self.compute_adv_loss(image_dis, self.image_adv_loss, None, fake_x)
        latent_adv_loss = self.compute_adv_loss(latent_dis, self.latent_adv_loss, None, fake_z)
        rec_loss = self.rec_loss(real_x, rec_x)

        if update_grads:
            for model_name, model, losses in [
                ('gen', gen, [('adv', image_adv_loss, 1),  ('rec', rec_loss, self.gen_rec_loss_weight)]),
                ('enc', enc, [('adv', latent_adv_loss, 1), ('rec', rec_loss, self.enc_rec_loss_weight)]),
            ]:
                for loss_name, loss, weight in losses:
                    grad_info = self.grad_norm_policy_helper.update_grad_info(model, model_name, loss, loss_name)

                    if logger is not None and n_iter is not None:
                        for layer_name, val in grad_info.items():
                            logger.add_scalars(
                                f'w_{model_name}_grad_info/grad_norm_of_{layer_name}',
                                {loss_name: weight * val}, n_iter)

            if self.use_grad_norm_policy:
                self.enc_rec_loss_weight = self.grad_norm_policy_helper.recompute_weight('enc',
                                                                                         loss_name='rec',
                                                                                         with_respect_to_loss_name='adv')
                self.gen_rec_loss_weight = self.grad_norm_policy_helper.recompute_weight('gen',
                                                                                         loss_name='rec',
                                                                                         with_respect_to_loss_name='adv')

                if logger is not None and n_iter is not None:
                    logger.add_scalar('rec_loss_weight_for_encoder', self.enc_rec_loss_weight, n_iter)
                    logger.add_scalar('rec_loss_weight_for_generator', self.gen_rec_loss_weight, n_iter)

        if update_parameters:
            self.enc_gen_opt.zero_grad()

            # 1. Backpropagate rec loss
            rec_loss.backward(retain_graph=True)

            # multiply by weight for encoder and generator
            for p in enc.parameters():
                if p.requires_grad:
                    p.grad *= self.enc_rec_loss_weight

            for p in gen.parameters():
                if p.requires_grad:
                    p.grad *= self.gen_rec_loss_weight

            # 2. Backpropagate other losses
            (latent_adv_loss + image_adv_loss).backward()
            self.enc_gen_opt.step()

        loss_info = {
            'image_adv_loss': image_adv_loss.item(),
            'latent_adv_loss': latent_adv_loss.item(),
            'rec_loss': rec_loss.item()
        }
        return loss_info


class GradientNormalizingWeightPolicyHelper:
    def __init__(self, max_history_len=200, alpha=0.05, use_last_n_for_average=200):
        self._max_history_len = max_history_len
        self._alpha = alpha
        self._use_last_n_for_average = use_last_n_for_average
        self.grads = \
            defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: []
                    )
                )
            )

    def update_grad_info(self, model, model_name, loss, loss_name):
        model.zero_grad()
        loss.backward(retain_graph=True)

        grad_info = {}
        for layer_name, value in model.named_parameters():
            layer_name = layer_name.replace('model.', '').replace('.', '/')
            if (re.match(r'.*weight$', layer_name) is not None) and (value.grad is not None):
                # it's better to use np.linalg.norm(value.grad.data.cpu().numpy()) here
                grad_norm = value.grad.data.cpu().numpy().std()
                self.grads[model_name][loss_name][layer_name].append(grad_norm)

                # delete old values (take only "max_history_len" last values)
                self.grads[model_name][loss_name][layer_name] = self.grads[model_name][loss_name][layer_name][-self._max_history_len:]

                grad_info[layer_name] = grad_norm
        model.zero_grad()
        return grad_info

    def recompute_weight(self, model_name, loss_name, with_respect_to_loss_name):
        weights_per_layer = []

        for layer_name in self.grads[model_name][loss_name].keys():
            # smooth out the noise
            loss_smoothed_vals = ewma_vectorized(self.grads[model_name][loss_name][layer_name], self._alpha)
            respect_to_loss_smoothed_vals = ewma_vectorized(self.grads[model_name][with_respect_to_loss_name][layer_name], self._alpha)

            weights = respect_to_loss_smoothed_vals / loss_smoothed_vals

            # delete old values (take only "use_last_n_for_average" last values)
            weights = weights[-self._use_last_n_for_average:]
            weights_per_layer.extend(weights)

        return np.median(weights_per_layer)
