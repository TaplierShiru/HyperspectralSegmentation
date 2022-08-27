#!/usr/bin/env python
# coding: utf-8
# %%

import os

GPU_ID = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

import sys
sys.path.append('/home/rustam/hyperspecter_segmentation/makitorch')
sys.path.append('/home/rustam/hyperspecter_segmentation/')

PREFIX_INFO_PATH = '/home/rustam/hyperspecter_segmentation/danil_cave/kfolds_data/kfold0'
PATH_DATA = '/raid/rustam/hyperspectral_dataset/new_cropped_hsi_data'


from multiprocessing.dummy import Pool
import copy
import glob
from makitorch import *
import math
import numpy as np
import numba as nb
import comet_ml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torchvision import utils
import cv2
from Losses import FocalLoss
import matplotlib.pyplot as plt

import seaborn as sns
import json
from tqdm import tqdm

from sklearn.decomposition import PCA
from makitorch.architectures.U2Net import U2Net

from hsi_dataset_api import HsiDataset

from makitorch.dataloaders.HsiDataloader import HsiDataloader
from makitorch.architectures.Unet import Unet, UnetWithFeatureSelection
from makitorch.loss import muti_bce_loss_fusion
from sklearn.metrics import jaccard_score
np.set_printoptions(suppress=True)
torch.backends.cudnn.benchmark = True # Speed up

from makitorch.data_tools.augmentation import DataAugmentator
from makitorch.data_tools.augmentation import BaseDataAugmentor
from makitorch.data_tools.preprocessing import BaseDataPreprocessor
from makitorch.data_tools.preprocessing import DataPreprocessor

from typing import Callable, Optional, Union

import torch
from sklearn.utils import shuffle
from hsi_dataset_api import HsiDataset


@nb.njit
def cut_into_parts(
        image: np.ndarray, mask: np.ndarray, 
        shift_h: int, shift_w: int):
    image_parts_list = []
    mask_parts_list = []

    for h_i in range(shift_h, mask.shape[0]-shift_h):
        for w_i in range(shift_w, mask.shape[1]-shift_w):
            img_part = image[:, 
                (h_i - shift_h): (h_i + shift_h + 1), 
                (w_i - shift_w): (w_i + shift_w + 1)
            ]
            mask_part = mask[
                (h_i - shift_h): (h_i + shift_h + 1), 
                (w_i - shift_w): (w_i + shift_w + 1)
            ]

            image_parts_list.append(img_part)
            mask_parts_list.append(mask_part)
    return image_parts_list, mask_parts_list



class ShmData:

    def __init__(self, shm_name, shape, dtype):
        self.shm_name = shm_name
        self.shape = shape
        self.dtype = dtype


class DatasetCreator:

    def __init__(
            self, 
            data_path: str,
            indices = None,
            cut_window=(8, 8),
            map_mask_to_class=False,
            shuffle_then_prepared=False):
        self.dataset = HsiDataset(data_path)
        self.cut_window = cut_window
        self.map_mask_to_class = map_mask_to_class
        
        self.images = []
        self.masks = []
        self._shm_imgs = None
        self._shm_masks = None

        for idx, data_point in tqdm(enumerate(self.dataset.data_iterator(opened=True, shuffle=False))):
            if indices is not None and idx not in indices:
                continue
            image, mask = data_point.hsi, data_point.mask
            if cut_window is not None:
                image_parts, mask_parts = self._cut_with_window(image, mask, cut_window)
                self.images += image_parts
                self.masks += mask_parts
            else:
                self.images.append(image)
                self.masks.append(mask)
        if shuffle_then_prepared:
            self.images, self.masks = shuffle(self.images, self.masks)
                    
    
    def _cut_with_window(self, image, mask, cut_window):
        assert len(cut_window) == 2
        h_win, w_win = cut_window
        _, h, w = image.shape
        # Append values to mask
        mask_shape = list(mask.shape)
        # Calculate additional padding values
        shift_w, shift_h = (
            (w_win - 1)//2, 
            (h_win - 1)//2
        )
        if shift_w == 0 and shift_h == 0:
            padding_img = image
            padding_mask = mask
        else:
            # Multiply on 2
            mask_shape[0] += shift_h * 2
            mask_shape[1] += shift_w * 2
            # Create padding mask and copy original data
            padding_mask = np.zeros(mask_shape, dtype=mask.dtype)
            padding_mask[shift_h:-shift_h, shift_w:-shift_w] = mask
            # Create padding img and copy original data
            img_shape = list(image.shape)
            img_shape[1] += shift_h * 2
            img_shape[2] += shift_w * 2
            padding_img = np.zeros(img_shape, dtype=image.dtype)
            padding_img[:, shift_h:-shift_h, shift_w:-shift_w] = image
        return cut_into_parts(
            image=padding_img, mask=padding_mask,
            shift_w=shift_w, shift_h=shift_h
        )


class HsiDataloaderCutter(torch.utils.data.IterableDataset):
    def __init__(
            self, 
            images, masks,
            batch_size,
            augmentation: Optional[Union[DataAugmentator, Callable]] = BaseDataAugmentor(),
            preprocessing: Optional[Union[DataPreprocessor, Callable]] = None,
            shuffle_data=False,
            cut_window=(8, 8),
            map_mask_to_class=False,
        ):
        super().__init__()
        self.shuffle_data = shuffle_data
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        if cut_window is not None and cut_window[0] == 1 and cut_window[1] == 1:
            self.ignore_image_augs = True
        else:
            self.ignore_image_augs = False
        self.cut_window = cut_window
        self.map_mask_to_class = map_mask_to_class

        self.images = images
        self.masks = masks

    def __iter__(self):
        assert self.images is not None and self.masks is not None
        if self.shuffle_data:
            self.images, self.masks = shuffle(self.images, self.masks)
        
        batch_parts = len(self.images) // self.batch_size

        for i in range(batch_parts):
            image_batch = self.images[i * self.batch_size: (i + 1) * self.batch_size]
            mask_batch = self.masks[i * self.batch_size: (i + 1) * self.batch_size] 

            image_batch, mask_batch = self.preprocessing(
                image_batch, mask_batch,
                map_mask_to_class=self.map_mask_to_class
            )
            yield self.augmentation(
                image_batch, mask_batch, 
                map_mask_to_class=self.map_mask_to_class,
                ignore_image_augs=self.ignore_image_augs
            )


def clear_metric_calculation(final_metric, target_t, pred_t, num_classes=17, using_pred=False):
    """
    
    Parameters
    ----------
    final_metric: torch.Tensor
        Tensor with shape (N, C)
    target_t: torch.Tensor or list
        Tensor with shape (N, 1, H, W)
    pred_t: torch.Tensor or list
        Tensor with shape (N, 1, H, W)
    
    """
    # For each image
    final_metric_dict = dict([
        (str(i), []) for i in range(num_classes)
    ])
    for metric_s, target_t_s, pred_t_s in zip(final_metric, target_t, pred_t):
        unique_indx_target = torch.unique(target_t_s.long())
        unique_indx_pred = None
        if using_pred:
            if isinstance(pred_t_s, np.ndarray):
                pred_t_s = torch.from_numpy(pred_t_s)
            unique_indx_pred = torch.unique(pred_t_s.long())
        for i in range(num_classes):
            if i in unique_indx_target or \
                    (unique_indx_pred is not None and i in unique_indx_pred):
                final_metric_dict[str(i)].append(metric_s[i])
    
    mean_per_class_metric = [
        sum(final_metric_dict[str(i)]) / len(final_metric_dict[str(i)])
        if len(final_metric_dict[str(i)]) != 0
        else 0.0
        for i in range(num_classes)
    ] 
    mean_metric = sum(mean_per_class_metric) / len(mean_per_class_metric)
    return mean_per_class_metric, mean_metric


def cut_into_parts_model_input(
        image: torch.tensor,
        shift_w: int, shift_h: int):
    image_parts_list = []
    for h_i in range(shift_h, image.shape[2]-shift_h):
        for w_i in range(shift_w, image.shape[3]-shift_w):
            img_part = image[:, :,
                (h_i - shift_h): (h_i + shift_h + 1), 
                (w_i - shift_w): (w_i + shift_w + 1)
            ]
            image_parts_list.append(img_part)
    return image_parts_list


def merge_parts_into_single_mask(preds, shape):
    pred_mask = torch.zeros(
        shape,
        dtype=preds.dtype, device=preds.device
    )
    counter = 0

    for h_i in range(shape[2]):
        for w_i in range(shape[3]):
            # Map preds, from (17,) -> (17, 1, 1) - aka pixel result
            pred_mask[
                :, :, 
                h_i:(h_i + 1), w_i:(w_i + 1)
            ] = preds[counter].unsqueeze(dim=-1).unsqueeze(dim=-1)
            counter += 1
    return pred_mask


def collect_prediction_and_target(
        eval_loader, model, cut_window=(8, 8), 
        image_shape=(512, 512), num_classes=17,
        divided_batch=512):
    target_list = []
    pred_list = []
    
    for in_data_x, val_data in iter(eval_loader):
        batch_size = in_data_x.shape[0]
        # We will cut image into peases and stack it into single BIG batch
        h_win, w_win = cut_window
        # Calculate additional padding values
        shift_w, shift_h = (
            (w_win - 1)//2, 
            (h_win - 1)//2
        )
        if shift_w == 0 and shift_h == 0:
            padding_img = in_data_x
        else:
            # Create padding img and copy original data
            img_shape = list(in_data_x.shape)
            img_shape[2] += shift_h * 2
            img_shape[3] += shift_w * 2
            padding_img = torch.zeros(img_shape, dtype=in_data_x.dtype, device=in_data_x.device)
            padding_img[:, :, shift_h:-shift_h, shift_w:-shift_w] = in_data_x
            padding_img = padding_img
        in_data_x_parts_list = cut_into_parts_model_input(
            padding_img, 
            shift_w=shift_w, shift_h=shift_h
        )
        in_data_x_batch = torch.cat(in_data_x_parts_list, dim=0) # (image_shape[0] * image_shape[1], 17, 8, 8)
        part_divided = len(in_data_x_batch) // divided_batch
        pred_batch_list = []
        for b_i in range(part_divided):
            if b_i == (part_divided - 1):
                # last
                single_batch = in_data_x_batch[b_i * divided_batch:]
            else:
                single_batch = in_data_x_batch[b_i * divided_batch: (b_i+1) * divided_batch]
            # Make predictions
            preds = model(single_batch) # (divided_batch, num_classes)
            pred_batch_list.append(preds)
        preds = torch.cat(pred_batch_list, dim=0) 
        # Create full image again from peases
        pred_mask = merge_parts_into_single_mask(
            preds=preds, 
            shape=(batch_size, num_classes, image_shape[0], image_shape[1]), 
        )
        target_list.append(val_data)
        pred_list.append(pred_mask)
    return (torch.cat(pred_list, dim=0), 
            torch.cat(target_list, dim=0)
    )
        

def calculate_iou(pred_list, target_list, num_classes=17):
    res_list = []
    pred_as_mask_list = []
    
    for preds, target in zip(pred_list, target_list):
        # preds - (num_classes, H, W)
        preds = preds.detach()
        # target - (H, W)
        target = target.detach()

        preds = nn.functional.softmax(preds, dim=0)
        preds = torch.argmax(preds, dim=0)
        pred_as_mask_list.append(preds)
        
        preds_one_hoted = torch.nn.functional.one_hot(preds, num_classes).view(-1, num_classes).cpu()
        target_one_hoted = torch.nn.functional.one_hot(target, num_classes).view(-1, num_classes).cpu()
        res = jaccard_score(target_one_hoted, preds_one_hoted, average=None, zero_division=1)
        res_list.append(
            res
        )
    
    res_np = np.stack(res_list)
    #res_np = res_np.mean(axis=0)
    return res_np, pred_as_mask_list


def dice_loss(preds, ground_truth, eps=1e-5, dim=None, use_softmax=False, softmax_dim=1):
    """
    Computes Dice loss according to the formula from:
    V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation
    Link to the paper: http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf
    Parameters
    ----------
    preds : tf.Tensor
        Predicted probabilities.
    ground_truth : tf.Tensor
        Ground truth labels.
    eps : float
        Used to prevent division by zero in the Dice denominator.
    axes : list
        Defines which axes the dice value will be computed on. The computed dice values will be averaged
        along the remaining axes. If None, Dice is computed on an entire batch.
    Returns
    -------
    tf.Tensor
        Scalar dice loss tensor.
    """
    ground_truth = ground_truth.float().to(device=preds.device)
    
    if use_softmax:
        preds = nn.functional.softmax(preds, dim=softmax_dim)
    
    numerator = preds * ground_truth
    numerator = torch.sum(numerator, dim=dim)

    p_squared = torch.square(preds)
    p_squared = torch.sum(p_squared, dim=dim)
    # ground_truth is not squared to avoid unnecessary computation.
    # 0^2 = 0
    # 1^2 = 1
    g_squared = torch.sum(torch.square(ground_truth), dim=dim)
    denominator = p_squared + g_squared + eps

    dice = 2 * numerator / denominator
    return 1 - dice


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if getattr(m, 'bias') is not None:
            m.bias.data.fill_(0.01)


class NnModel(pl.LightningModule):
    def __init__(
            self, model, loss,
            T_0=10, T_mult=2, experiment=None, enable_image_logging=True,
            cut_window=(8, 8), lr=1e-3, lr_list=None, epoch_list=None):
        super().__init__()
        self.model = model
        self.loss = loss
        self.cut_window = cut_window
        self.lr = lr
        self.experiment = experiment
        self.enable_image_logging = enable_image_logging

        if lr_list is not None and epoch_list is not None:
            if len(lr_list) != len(epoch_list):
                raise ValueError(
                    f"lr_list={lr_list} and epoch_list={epoch_list}"+\
                    " must be arrays of same length"
                )
            print(
                f'Using dynamic lr with next setup: lr_list={lr_list} epoch_list={sorted(epoch_list)}.\n+'+\
                'Epoch list is sorted by increasing.'
            )
            # Further lr/epoch value will be added/deleted into list
            # So, make copy to make sure that original data is safe
            lr_list = copy.deepcopy(lr_list)
            epoch_list = copy.deepcopy(epoch_list)

            self.lr_list = lr_list
            self.epoch_list = sorted(epoch_list)
            self.is_lr_must_change = True
        else:
            self.is_lr_must_change = False

        self.T_0 = T_0
        self.T_mult = T_mult
            
    def forward(self, x):
        out = self.model(x)
        return out
    
    def on_train_epoch_start(self):
        if self.is_lr_must_change and len(self.epoch_list) != 0:
            # Check, current epoch bigger than epoch on which must be updated lr
            if self.current_epoch >= self.epoch_list[0]:
                # Update optimizer
                self.lr = self.lr_list[0]
                self.trainer.accelerator.setup_optimizers(self.trainer)
                # Clear used variables
                del self.epoch_list[0]
                del self.lr_list[0]

    def configure_optimizers(self):
        # Change lr after some epoch
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3095
        # In this code I do something different and its more what I like
        print(f'Init optimizer with params: lr={self.lr}, T_0={self.T_0}, T_mult={self.T_mult}')
        optimizer = optim.Adam( 
            self.parameters(), lr=self.lr
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.T_0, T_mult=self.T_mult, eta_min=0
        )
        return { "optimizer": optimizer, "lr_scheduler": lr_scheduler }
    
    def training_step(self, train_batch, batch_idx):
        img, mask = train_batch
        preds = self.model(img) # (N, C, 8, 8)
        loss = self.loss(preds, mask) # (N, 8, 8)
        self.log('train_loss', loss)
        if self.experiment is not None:
            self.experiment.log_metric("train_loss", loss, epoch=self.current_epoch, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        return batch
    
    def validation_epoch_end(self, outputs):
        print('Size epoch end input: ', len(outputs))
        
        pred_tensor, target_tensor = collect_prediction_and_target(outputs, self.model, cut_window=self.cut_window)
        target_one_hotted_tensor = torch.nn.functional.one_hot(
            target_tensor, 17 # Num classes
        )
        # (N, H, W, C) --> (N, C, H, W)
        target_one_hotted_tensor = target_one_hotted_tensor.permute(0, -1, 1, 2)
        dice_loss_val = dice_loss(pred_tensor, target_one_hotted_tensor, dim=[0, 2, 3], use_softmax=True, softmax_dim=1)
        metric, pred_as_mask_list = calculate_iou(pred_tensor, target_tensor)
        
        for batch_idx, (metric_s, target_s, pred_s) in enumerate(zip(metric, target_tensor, pred_as_mask_list)):
            if self.enable_image_logging:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                sns.heatmap(pred_s.cpu().detach().numpy(), ax=ax1, vmin=0, vmax=17)
                sns.heatmap(target_s.cpu().detach().numpy(), ax=ax2, vmin=0, vmax=17)
                fig.savefig(f'temp_fig_{GPU_ID}.png')
                plt.close(fig)

                if self.experiment is not None:
                    # For Comet logger
                    self.experiment.log_image(
                        f'temp_fig_{GPU_ID}.png', name=f'{batch_idx}', 
                        overwrite=False, step=self.global_step
                    )
            
            d = {f'iou_{i}': iou for i, iou in enumerate(metric_s)}
            
            if self.experiment is not None:
                self.experiment.log_metrics(d, epoch=self.current_epoch)
            else:
                print(d)
        if self.experiment is not None:
            # Add confuse matrix
            self.experiment.log_confusion_matrix(
                target_tensor.cpu().detach().numpy().reshape(-1), 
                torch.stack(
                    [elem.cpu() for elem in pred_as_mask_list], 
                    dim=0
                ).cpu().detach().numpy().reshape(-1)
            )
            
        mean_per_class_metric, mean_metric = clear_metric_calculation(
            metric, target_tensor, pred_as_mask_list
        )
        mean_dice_loss_per_class_dict = {
            f"mean_dice_loss_per_class_{i}": d_l.float()
            for i, d_l in enumerate(dice_loss_val)
        }
        mean_dice_loss_dict = {
            f"mean_dice_loss": dice_loss_val.mean().float()
        }
        mean_iou_class_dict = {
            f"mean_iou_class_{i}": torch.tensor(iou, dtype=torch.float)
            for i, iou in enumerate(mean_per_class_metric)
        }
        mean_iou_dict = {
            "mean_iou": torch.tensor(mean_metric, dtype=torch.float),
        }
        
        # Log this metric in order to save checkpoint of experements
        self.log_dict(mean_iou_dict)
        
        if self.experiment is not None:
        
            self.experiment.log_metrics(
                mean_dice_loss_per_class_dict,
                epoch=self.current_epoch
            )

            self.experiment.log_metrics(
                mean_dice_loss_dict,
                epoch=self.current_epoch
            )

            self.experiment.log_metrics(
                mean_iou_class_dict,
                epoch=self.current_epoch
            )

            self.experiment.log_metrics(
                mean_iou_dict,
                epoch=self.current_epoch
            )
        else:
            print(mean_dice_loss_per_class_dict)
            print(mean_dice_loss_dict)
            print(mean_iou_class_dict)
            print(mean_iou_dict)
            print('---------------------------------')


device = 'cuda:0'
pca_explained_variance = np.load(f'{PREFIX_INFO_PATH}/kfold0_PcaExplainedVariance_.npy')
pca_mean = np.load(f'{PREFIX_INFO_PATH}/kfold0_PcaMean.npy')
pca_components = np.load(f'{PREFIX_INFO_PATH}/kfold0_PcaComponents.npy')


def pca_transformation(x):
    if len(x.shape) == 3:
        x_t = x.reshape((x.shape[0], -1)) # (C, H, W) -> (C, H * W)
        x_t = np.transpose(x_t, (1, 0)) # (C, H * W) -> (H * W, C)
        x_t = x_t - pca_mean
        x_t = np.dot(x_t, pca_components.T) / np.sqrt(pca_explained_variance)
        return x_t.reshape((x.shape[1], x.shape[2], pca_components.shape[0])).astype(np.float32, copy=False) # (H, W, N)
    elif len(x.shape) == 4:
        # x - (N, C, H, W)
        x_t = np.transpose(x, (0, 2, 3, 1)) # (N, C, H, W) -> (N, H, W, C)
        x_t = x_t - pca_mean
        x_t = np.dot(x_t, pca_components.T) / np.sqrt(pca_explained_variance)
        x_t = np.transpose(x_t, (0, -1, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
        return x_t.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown shape={x.shape}, must be of len 3 or 4.")

def standartization(img, mean, std):
    img -= mean
    img /= std
    return img

def standartization_pool(mean, std):
    # X shape - (N, C, H, W)
    # from shape (comp,) -> (1, comp, 1, 1)
    mean = np.expand_dims(np.expand_dims(np.array(mean, dtype=np.float32), axis=-1), axis=-1)
    std = np.expand_dims(np.expand_dims(np.array(std, dtype=np.float32), axis=-1), axis=-1)
    
    return lambda x: standartization(x, mean=mean, std=std)


@nb.njit
def mask2class_np(mask_np):
    new_class_np = np.zeros((len(mask_np), 1), dtype=mask_np.dtype)
    for m_i in range(len(mask_np)):
        single_mask = mask_np[m_i]
        # (1, h, w)
        shape = single_mask.shape
        shift_w, shift_h = (shape[2]-1)//2, (shape[1]-1)//2

        new_class_np[m_i] = single_mask[0, shift_h, shift_w]
    return new_class_np


def mask2class_single(mask):
    # (1, h, w)
    shape = mask.shape
    shift_w, shift_h = (shape[2]-1)//2, (shape[1]-1)//2
    return mask[0:1, shift_h, shift_w]


with open(f'{PREFIX_INFO_PATH}/data_standartization_params_kfold0.json', 'r') as f:
    data_standartization_params = json.load(f)
MEAN = data_standartization_params.get('means')
STD = data_standartization_params.get('stds')

MEAN_EXPAND = np.expand_dims(np.expand_dims(MEAN, axis=-1), axis=-1)
STD_EXPAND = np.expand_dims(np.expand_dims(STD, axis=-1), axis=-1)


def preprocessing(imgs, masks, map_mask_to_class=False):
    imgs_np = np.asarray(imgs, dtype=np.float32) # (N, 237, 1, 1)
    masks_np = np.asarray(masks, dtype=np.int64) # (N, 1, 1, 3)
    # Wo PCA
    # _images = [np.transpose(image, (1, 2, 0)) for image in imgs]
    # W Pca
    _images = pca_transformation(imgs_np)
    _images = standartization(_images, MEAN_EXPAND, STD_EXPAND)
    _masks = np.transpose(masks_np[..., 0:1], (0, -1, 1, 2))
    if map_mask_to_class:
        _masks = mask2class_np(_masks)
    return _images, _masks


def test_augmentation(image, mask, map_mask_to_class=False, **kwargs):
    image = torch.from_numpy(image)
    image = image.float()
    #image = (image - image.min()) / (image.max() - image.min())
    
    mask = torch.from_numpy(mask)
    mask = mask.long()
    mask = torch.squeeze(mask, 1) # (N, C, H, W), where C = 1, squeeze it
    return image, mask



def aug_random_rotate(image, mask, map_mask_to_class=False, **kwargs):
    angle = T.RandomRotation.get_params((-30, 30))
    image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
    if not map_mask_to_class:
        mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
    return image, mask


def aug_flip_horizontal(image, mask, map_mask_to_class=False, **kwargs):
    if torch.rand(1) > 0.5:
        image = TF.hflip(image)
        if not map_mask_to_class:
            mask = TF.hflip(mask)
    return image, mask


def aug_flip_vertical(image, mask, map_mask_to_class=False, **kwargs):
    if torch.rand(1) > 0.5:
        image = TF.vflip(image)
        if not map_mask_to_class:
            mask = TF.vflip(mask)
    return image, mask


MASK_AUG_SCALE = 100
MASK_AUG_COMPARE = 90
RandomEraseTorch = T.RandomErasing(
    p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), 
    value='random', inplace=False
)
def aug_random_erase(image, mask, map_mask_to_class=False, **kwargs):
    # Create mask in order to take area of aug
    mask_aug_area = torch.ones(
        1, image.shape[1], image.shape[2], 
        dtype=image.dtype, device=image.device
    ) * MASK_AUG_SCALE
    in_x = torch.cat([image, mask_aug_area], dim=0)
    # Apply aug
    img_aug = RandomEraseTorch(in_x)
    image = img_aug[:-1]
    if not map_mask_to_class:
        mask_aug = img_aug[-1:]
        # Take mask and reverse values
        # 0 - cutout zone, 1 - good zone
        mask_aug = (mask_aug > MASK_AUG_COMPARE).long()
        mask = mask * mask_aug
    return image, mask


AUGS_LIST = [
    aug_random_rotate,
    aug_flip_horizontal,
    aug_flip_vertical,
#    aug_random_erase
]


def augmentation(image, mask, map_mask_to_class=False, ignore_image_augs=False):
    image = torch.from_numpy(image)
    image = image.float()
    mask = torch.from_numpy(mask)
    mask = mask.long()

    if not ignore_image_augs:
        for aug_func in AUGS_LIST:
            image, mask = aug_func(image, mask, map_mask_to_class=map_mask_to_class)
    
    #image = (image - image.min()) / (image.max() - image.min())
    mask = torch.squeeze(mask, -1)
    return image, mask


class MySuperNetLittleInput(nn.Module):
    
    def __init__(self, in_f=237, out_f=17, cut_window=(8, 8), *args):
        super().__init__()
        #self.bn_start = nn.BatchNorm3d(in_f)
        
        self.conv1 = nn.Conv2d(in_f, 128, kernel_size=3, stride=1, padding=1)
        # (N, 128, 8, 8)
        self.bn1 = nn.BatchNorm2d(128)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # (N, 128, 8, 8)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # (N, 64, 8, 8)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # (N, 64, 8, 8)
        self.bn4 = nn.BatchNorm2d(64)
        self.act4 = nn.ReLU()

        final_h, final_w = cut_window[0], cut_window[1]
        self.f1 = nn.Linear(64 * final_h * final_w, 256, bias=False)
        self.dropout1 = nn.Dropout(0.4)
        self.f2 = nn.Linear(256, out_f, bias=False)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        x = torch.flatten(x, 1) # aka (N, -1)
        x = self.f1(x)
        x = self.dropout1(x)

        x = self.f2(x)
        return x

