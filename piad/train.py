import os
import tqdm
import torch
import numpy as np
import random
import argparse
import yaml
from torch.utils.data import DataLoader

from piad.latent_model import GaussianNoise
from piad.logger import Logger
from piad.networks import ResNetGenerator, ResNetEncoder, LatentDiscriminator
from piad.datasets import create_dataset, create_transform, inf_dataloader
from piad.optimizer import Optimizer
from piad.adv_losses import WassersteinLoss
from piad.rec_losses import PerceptualLoss


class PIADTrainer:
    def __init__(self, config):
        self.config = config

        if config['random_seed'] is not None:
            torch.manual_seed(config['random_seed'])
            torch.cuda.manual_seed(config['random_seed'])
            np.random.seed(config['random_seed'])
            random.seed(config['random_seed'])
            torch.backends.cudnn.deterministic = True

        self.verbose = config['verbose']
        output_root = os.path.join(config['output_root'], str(config["normal_class"]), str(config["run"]))
        self.checkpoint_root = os.path.join(output_root, 'checkpoint')
        self.log_root = os.path.join(output_root, 'logs')

        self.logger = Logger(self.log_root)

        self.image_res = config['image_res']
        self.image_dim = config['image_dim']
        self.latent_res = config['latent_res']
        self.latent_dim = config['latent_dim']

        self.iters = config['iters']
        self.log_iter = config['log_iter']
        self.checkpoint_iter = config['checkpoint_iter']
        self.image_sample_iter = config['image_sample_iter']
        self.update_grad_norm_iter = config.get('update_grad_norm_iter', 500)

        self.batch_size = config['batch_size']
        self.n_dis = config['n_dis']
        assert self.n_dis >= 1

        "=========================================== create data model ================================================"

        dataset_type = config['dataset_type']
        dataset_root = config['dataset_root']
        transform = create_transform(dataset_type)
        train_dataset = create_dataset(dataset_type, dataset_root, 'train', config["normal_class"], normal=True,
                                       transform=transform, extra_dataset_params=config.get("extra_dataset_params"))
        self.image_model = inf_dataloader(
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                       num_workers=config['num_workers'])
        )

        "=========================================== create latent model =============================================="

        self.latent_model = GaussianNoise(self.latent_dim, self.latent_res, self.batch_size)

        "============================================= create networks ================================================"

        self.gen = ResNetGenerator(self.image_res, self.image_dim, self.latent_res, self.latent_dim, config['gen']).cuda()
        self.enc = ResNetEncoder(self.image_res, self.image_dim, self.latent_res, self.latent_dim, config['enc']).cuda()
        self.image_dis = ResNetEncoder(self.image_res, self.image_dim, 1, 1, config['image_dis'], use_mbstddev=True).cuda()
        self.latent_dis = LatentDiscriminator(
            input_dim=self.latent_dim * self.latent_res * self.latent_res,
            inner_dims=config['latent_dis']).cuda()

        if self.verbose:
            print("====================== Generator ===========================")
            print(self.gen)
            print("====================== Image discriminator ===================")
            print(self.image_dis)
            print("====================== Encoder =============================")
            print(self.enc)
            print("====================== Latent discriminator================")
            print(self.latent_dis)

        "=========================================== create losses ===================================================="

        image_adv_loss = WassersteinLoss(**config['wasserstein_loss_kwargs'])
        latent_adv_loss = WassersteinLoss(**config['wasserstein_loss_kwargs'])
        res_loss = PerceptualLoss(**config['perceptual_loss_kwargs'], reduction='mean')

        "=========================================== create optimizers ================================================"

        self.optimizer = Optimizer(
            gen_params=self.gen.parameters(),
            enc_params=self.enc.parameters(),
            image_dis_params=self.image_dis.parameters(),
            latent_dis_params=self.latent_dis.parameters(),
            image_adv_loss=image_adv_loss,
            latent_adv_loss=latent_adv_loss,
            rec_loss=res_loss,
            adam_kwargs=config['adam_kwargs'],
            enc_rec_loss_weight=config['enc_rec_loss_weight'],
            gen_rec_loss_weight=config['gen_rec_loss_weight'],
            use_grad_norm_policy=config['use_grad_norm_policy']
        ).cuda()

        "=========================================== data for logging ================================================="

        self.n_image_for_visualization = 5
        self.display_z = next(self.latent_model)[:self.n_image_for_visualization].cpu().detach()
        self.display_x = next(self.image_model)[:self.n_image_for_visualization].cpu().detach()

        "=========================================== initialize ======================================================="

        self.tqdm_logger = tqdm.tqdm(total=self.iters)

        if config['finetune_from'] is not None:
            self.load_state(torch.load(config['finetune_from']))

    def train(self):
        while self.tqdm_logger.n < self.tqdm_logger.total:
            self.tqdm_logger.update(1)

            "=========================================== train step ==================================================="

            for _ in range(self.n_dis):
                real_z = next(self.latent_model).cuda()
                real_x = next(self.image_model).cuda()
                fake_x = self.gen(real_z.detach())
                fake_z = self.enc(real_x)

                image_dis_losses = self.optimizer.compute_image_dis_loss(self.image_dis, real_x, fake_x, update_parameters=True)
                latent_dis_losses = self.optimizer.compute_latent_dis_loss(self.latent_dis, real_z, fake_z, update_parameters=True)

            real_z = next(self.latent_model).cuda()
            real_x = next(self.image_model).cuda()

            fake_x = self.gen(real_z)
            fake_z = self.enc(real_x)
            rec_x = self.gen(fake_z)

            gen_enc_losses = self.optimizer.compute_enc_gen_loss(
                self.enc, self.gen, self.latent_dis, self.image_dis,
                fake_x, real_x, fake_z, rec_x,
                update_parameters=True,
                logger=self.logger,
                n_iter=self.tqdm_logger.n,
                update_grads=self._if_time_to_react(self.update_grad_norm_iter)
            )

            del real_z, real_x, fake_z, fake_x, rec_x

            "============================================== logging ==================================================="

            if self._if_time_to_react(self.log_iter):
                self.logger.add_scalars('train/image_dis', image_dis_losses, self.tqdm_logger.n)
                self.logger.add_scalars('train/latent_dis', latent_dis_losses, self.tqdm_logger.n)
                self.logger.add_scalars('train/gen_enc', gen_enc_losses, self.tqdm_logger.n)

            if self._if_time_to_react(self.image_sample_iter):
                self._save_image_sample()

            "============================================ checkpoint =================================================="

            if self._if_time_to_react(self.checkpoint_iter):
                self._do_checkpoint()

        self._do_checkpoint()

    def get_state(self):
        return {
            'config': self.config,
            'n_iter': self.tqdm_logger.n,
            'gen': self.gen.state_dict(),
            'enc': self.enc.state_dict(),
            'latent_dis': self.latent_dis.state_dict(),
            'image_dis': self.image_dis.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'display_z': self.display_z.data,
            'display_x': self.display_x.data,
        }

    def load_state(self, state):
        self.display_z.data = state['display_z']
        self.display_x.data = state['display_x']

        self.gen.load_model_state(state['gen'])
        self.enc.load_model_state(state['enc'])
        self.latent_dis.load_model_state(state['latent_dis'])
        self.image_dis.load_model_state(state['image_dis'])
        self.optimizer.load_opt_state(state['optimizer'])

        self.tqdm_logger.update(state['n_iter'])

    def _if_time_to_react(self, every_iter):
        return self.tqdm_logger.n % every_iter == 0

    def _do_checkpoint(self):
        os.makedirs(self.checkpoint_root, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_root, f'iter_{self.tqdm_logger.n:07d}.tar')
        torch.save(self.get_state(), checkpoint_path)

        checkpoint_path = os.path.join(self.checkpoint_root, 'latest.tar')
        torch.save(self.get_state(), checkpoint_path)

    def _save_image_sample(self):
        torch.set_grad_enabled(False)
        self.optimizer.eval()

        images = torch.cat([
            self.gen(self.display_z.cuda()).cpu().detach(),
            self.display_x,
            self.gen(self.enc(self.display_x.cuda())).cpu().detach()
        ], 0)

        name = f'iter_{self.tqdm_logger.n:07d}.png'
        self.logger.save_image_samples(images, grid_size=self.image_res * self.n_image_for_visualization,
                                       name=name, nrow=self.n_image_for_visualization)
        self.logger.save_image_samples(images, grid_size=self.image_res * self.n_image_for_visualization,
                                       name='sample.png', nrow=self.n_image_for_visualization)

        torch.set_grad_enabled(True)
        self.optimizer.train()


def train(config, normal_class, run):
    config['normal_class'] = normal_class
    config['run'] = run

    trainer = PIADTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=str, help='Path to config')
    parser.add_argument('normal_class', type=int, help='Normal class')
    parser.add_argument('run', type=int, help='# of run')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    train(config, args.normal_class, args.run)


if __name__ == '__main__':
    main()
