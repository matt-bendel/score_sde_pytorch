# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import controllable_generation
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import matplotlib.pyplot as plt
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
# import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from losses import get_optimizer
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
FLAGS = flags.FLAGS

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

class DataTransform:
  """
  Data Transformer for training U-Net models.
  """

  def __init__(self):
    """
    Args:
        mask_func (common.subsample.MaskFunc): A function that can create  a mask of
            appropriate shape.
        resolution (int): Resolution of the image.
        which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
        use_seed (bool): If true, this class computes a pseudo random number generator seed
            from the filename. This ensures that the same mask is used for all the slices of
            a given volume every time.
    """
    np.random.seed(0)

    arr = np.ones((256, 256))
    arr[256 // 4: 3 * 256 // 4, 256 // 4: 3 * 256 // 4] = 0
    self.mask = torch.tensor(np.reshape(arr, (256, 256)), dtype=torch.float).repeat(3, 1, 1)
    torch.save(self.mask, 'mast.pt')

  def __call__(self, gt_im):
    # mean = torch.tensor([0.5, 0.5, 0.5])
    # std = torch.tensor([0.5, 0.5, 0.5])
    gt = gt_im
    masked_im = gt * self.mask

    return gt.float(), masked_im.float(), self.mask.float()


def create_datasets():
  transform = transforms.Compose([transforms.ToTensor(), DataTransform()])
  dataset = datasets.ImageFolder('/storage/celebA-HQ/celeba_hq_256', transform=transform)
  train_data, dev_data, test_data = torch.utils.data.random_split(
    dataset, [27000, 2000, 1000],
    generator=torch.Generator().manual_seed(0)
  )

  return test_data, dev_data, train_data


def create_data_loaders():
  test_data, dev_data, train_data = create_datasets()

  train_loader = DataLoader(
    dataset=train_data,
    batch_size=128,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    drop_last=True,
  )

  dev_loader = DataLoader(
    dataset=dev_data,
    batch_size=128,
    num_workers=16,
    pin_memory=True,
    drop_last=True,
  )

  test_loader = DataLoader(
    dataset=test_data,
    batch_size=40,
    num_workers=16,
    pin_memory=True,
  )

  return train_loader, dev_loader, test_loader

def image_grid(x):
  size = config.data.image_size
  channels = config.data.num_channels
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def show_samples(x, tc):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.savefig(f'/storage/celebA-HQ/langevin_256_plots/samps_{tc}')
  plt.close()

def sample(config):
    b_size = 40
    _, _, test_ds = create_data_loaders()

    score_model = mutils.create_model(config)
    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer,
                 model=score_model, ema=ema)
    ckpt_filename = "/storage/celebA-HQ/weights/checkpoint_48.pth"
    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())

    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    predictor = ReverseDiffusionPredictor  # @param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
    corrector = LangevinCorrector  # @param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
    snr = 0.075  # @param {"type": "number"}
    n_steps = 1  # @param {"type": "integer"}
    probability_flow = False  # @param {"type": "boolean"}

    pc_inpainter = controllable_generation.get_pc_inpainter(sde,
                                                            predictor, corrector,
                                                            inverse_scaler,
                                                            snr=snr,
                                                            n_steps=n_steps,
                                                            probability_flow=probability_flow,
                                                            continuous=config.training.continuous,
                                                            denoise=True)

    total_count = 0

    num_samps = 32
    with torch.no_grad():
        for i, data in enumerate(test_ds):
            print(f"BATCH: {i+1}/{len(test_ds)}")
            batch, y, mask = data[0]
            batch = batch.cuda()
            mask = mask.cuda().repeat(batch.size(0)*num_samps, 1, 1, 1)

            super_batch = torch.zeros(batch.size(0)*num_samps, 3, 256, 256).cuda()

            for j in range(batch.size(0)):
                super_batch[j*num_samps:(j+1)*num_samps] = batch[j].unsqueeze(0).repeat(num_samps, 1, 1, 1)

            x = pc_inpainter(score_model, scaler(super_batch), mask)

            print("SAVING SAMPLES...")
            for j in range(batch.size(0)):
                samps = x[j*num_samps:(j+1)*num_samps, :, :, :]
                show_samples(samps, total_count)
                for k in range(num_samps):
                    save_dict = {
                        'gt': batch[j].cpu(),
                        'masked': y[j],
                        'x_hat': samps[k]
                    }
                    torch.save(save_dict, os.path.join('/storage/celebA-HQ/langevin_recons_256', f'image_{total_count}_sample_{k}.pt'))

                total_count += 1
            exit()
