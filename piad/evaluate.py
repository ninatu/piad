import os
import argparse
import yaml
import time
import torch
import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from piad.networks import ResNetGenerator, ResNetEncoder
from piad.rec_losses import PerceptualLoss
from piad.datasets import create_transform, create_dataset


def _compute_abnormality_scores(data_loader, loss, gen, enc, return_time=False):
    anomaly_scores = []
    times = []
    for images in tqdm.tqdm(data_loader):
        images = images.cuda()
        time_st = time.time()
        with torch.no_grad():
            reconstructed = gen(enc(images)).detach()
        anomaly_scores.extend(loss(images, reconstructed).detach().cpu().numpy())
        times.append(time.time() - time_st)
    if return_time:
        return anomaly_scores, times
    else:
        return anomaly_scores


def evaluate(config, normal_class, run, abnormal_class=None, batch_size=32):

    output_root = os.path.join(config['output_root'], str(normal_class), str(run))
    checkpoint_path = os.path.join(output_root, 'checkpoint', 'latest.tar')
    checkpoint = torch.load(checkpoint_path)

    dataset_type = config['dataset_type']
    dataset_root = config['dataset_root']
    num_workers = config['num_workers']
    transform = create_transform(dataset_type)
    normal_dataset = create_dataset(dataset_type, dataset_root, 'test', normal_class, normal=True,
                                    transform=transform, extra_dataset_params=config.get("extra_dataset_params"))
    anomaly_dataset = create_dataset(dataset_type, dataset_root, 'test', normal_class, normal=False,
                                     transform=transform, abnormal_class=abnormal_class,
                                     extra_dataset_params=config.get("extra_dataset_params"))
    normal_data_loader = DataLoader(normal_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size,
                                    drop_last=False)
    anomaly_data_loader = DataLoader(anomaly_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size,
                                     drop_last=False)

    gen = ResNetGenerator(config['image_res'], config['image_dim'], config['latent_res'], config['latent_dim'],
                          config['gen']).cuda()
    enc = ResNetEncoder(config['image_res'], config['image_dim'], config['latent_res'], config['latent_dim'],
                        config['enc']).cuda()
    gen.load_state_dict(checkpoint['gen'])
    enc.load_state_dict(checkpoint['enc'])

    res_loss = PerceptualLoss(**config['perceptual_loss_kwargs'], reduction='none').cuda()

    normal_preds, normal_times = _compute_abnormality_scores(normal_data_loader, res_loss, gen, enc, return_time=True)
    anomaly_preds, anomaly_times = _compute_abnormality_scores(anomaly_data_loader, res_loss, gen, enc,
                                                               return_time=True)

    y_true = np.concatenate((np.zeros_like(normal_preds), np.ones_like(anomaly_preds)))
    y_pred = np.concatenate((np.array(normal_preds), np.array(anomaly_preds)))

    roc_auc = roc_auc_score(y_true, y_pred)
    avg_time = np.array(normal_times + anomaly_times).mean()

    os.makedirs(output_root, exist_ok=True)
    results = pd.DataFrame([[roc_auc, avg_time * 1000]], columns=['ROC AUC', 'Time (ms)'])

    print("Model evaluation is complete. Results: ")
    print(results)

    result_filename = 'results.csv' if abnormal_class is None else f'{abnormal_class}.csv'
    results.to_csv(os.path.join(output_root, result_filename), index=False)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', help='Path to config')
    parser.add_argument('normal_class', type=int, help='Normal class')
    parser.add_argument('run', type=int, help='# of run')
    parser.add_argument('--abnormal_class', default=None,
                        help='Optional parameter used only in CelebA experiments. '
                             'Specify one of abnormal classes: '
                             '"Bald", "Mustache", "Bangs", "Eyeglasses", "Wearing_Hat"')
    parser.add_argument('--batch_size', default=32, help='batch_size')

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    evaluate(config, args.normal_class, args.run, args.abnormal_class, args.batch_size)


if __name__ == '__main__':
    main()
