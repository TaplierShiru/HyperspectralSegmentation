#!/usr/bin/env python
# coding: utf-8
# %%

import os

GPU_ID = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

import sys
sys.path.append('/home/rustam/hyperspecter_segmentation/makitorch')
sys.path.append('/home/rustam/hyperspecter_segmentation/')

PREFIX_INFO_PATH = '/home/rustam/hyperspecter_segmentation/danil_cave/kfolds_data/kfold0'
PATH_DATA = '/raid/rustam/hyperspectral_dataset/new_cropped_hsi_data'


from multiprocessing.dummy import Pool
from multiprocessing import shared_memory
import copy
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
        image: np.ndarray, mask: np.ndarray, h_parts: int, 
        w_parts: int, h_win: int, w_win: int):
    image_parts_list = []
    mask_parts_list = []

    for h_i in range(h_parts):
        for w_i in range(w_parts):
            img_part = image[:, 
                h_i * h_win: (h_i+1) * h_win, 
                w_i * w_win: (w_i+1) * w_win
            ]
            mask_part = mask[
                h_i * h_win: (h_i+1) * h_win, 
                w_i * w_win: (w_i+1) * w_win
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
            preprocessing: Optional[Union[DataPreprocessor, Callable]] = BaseDataPreprocessor(),
            indices = None,
            cut_window=(8, 8),
            map_mask_to_class=False,
            create_shared_memory=False,
            shuffle_then_prepared=False):
        self.dataset = HsiDataset(data_path)
        self.preprocessing = preprocessing
        self.cut_window = cut_window
        self.map_mask_to_class = map_mask_to_class
        self.create_shared_memory = create_shared_memory
        
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
        print("Preprocess data...")
        if self.preprocessing is not None:
            self.images, self.masks = self.preprocessing(
                self.images, self.masks, map_mask_to_class=map_mask_to_class
            )

        if shuffle_then_prepared:
            self.images, self.masks = shuffle(self.images, self.masks)

        # Create shared memory
        if create_shared_memory:
            print('Create shared memory...')
            # First - map images and masks into np
            self.images = np.asarray(self.images, dtype=np.float32)
            self.masks = np.asarray(self.masks, dtype=np.int64)
            # Imgs
            shm_imgs = shared_memory.SharedMemory(create=True, size=self.images.nbytes)
            shm_imgs_arr = np.ndarray(self.images.shape, dtype=self.images.dtype, buffer=shm_imgs.buf)
            shm_imgs_arr[:] = self.images[:]
            self.images = shm_imgs_arr # Do not keep dublicate 
            self.data_shm_imgs = ShmData(
                shm_name=shm_imgs.name, shape=self.images.shape, 
                dtype=self.images.dtype
            )
            self._shm_imgs = shm_imgs
            # Masks
            shm_masks = shared_memory.SharedMemory(create=True, size=self.masks.nbytes)
            shm_masks_arr = np.ndarray(self.masks.shape, dtype=self.masks.dtype, buffer=shm_masks.buf)
            shm_masks_arr[:] = self.masks[:]
            self.masks = shm_masks_arr # Do not keep dublicate 
            self.data_shm_masks = ShmData(
                shm_name=shm_masks.name, shape=self.masks.shape,
                dtype=self.masks.dtype
            )
            self._shm_masks = shm_masks
            print("Shared memory are created for imgs and masks!")

    def close_shm(self):
        if self.create_shared_memory:
            # Make sure there is no reference data to images/masks
            del self.images
            del self.masks
            self.images = []
            self.masks = []
            # Close and unlink
            if self._shm_masks is not None:
                self._shm_masks.close()
                self._shm_masks.unlink()
                self._shm_masks = None

            if self._shm_imgs is not None:
                self._shm_imgs.close()
                self._shm_imgs.unlink()
                self._shm_imgs = None
            print("Shared memory for masks and images are success cleared!")
                    
    
    def _cut_with_window(self, image, mask, cut_window):
        assert len(cut_window) == 2
        h_win, w_win = cut_window
        _, h, w = image.shape
        h_parts = h // h_win
        w_parts = w // w_win
        if h % h_win != 0:
            print(f"{h % h_win} pixels will be dropped by h axis. Input shape={image.shape}")

        if w % w_win != 0:
            print(f"{w % w_win} pixels will be dropped by w axis. Input shape={image.shape}")
        return cut_into_parts(
            image=image, mask=mask, h_parts=h_parts, w_parts=w_parts,
            h_win=h_win, w_win=w_win
        )


# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    shared_memory_imgs_data: ShmData = dataset.shared_memory_imgs_data
    shared_memory_masks_data: ShmData = dataset.shared_memory_masks_data
    if shared_memory_imgs_data is not None and shared_memory_masks_data is not None:
        # Take array from memory
        existing_shm_imgs = shared_memory.SharedMemory(name=shared_memory_imgs_data.shm_name)
        dataset_imgs_np = np.ndarray(
            shared_memory_imgs_data.shape, 
            dtype=shared_memory_imgs_data.dtype, buffer=existing_shm_imgs.buf
        )
        dataset.shm_imgs = existing_shm_imgs
        existing_shm_masks = shared_memory.SharedMemory(name=shared_memory_masks_data.shm_name)
        dataset_masks_np = np.ndarray(
            shared_memory_masks_data.shape, 
            dtype=shared_memory_masks_data.dtype, buffer=existing_shm_masks.buf
        )
        dataset.shm_masks = existing_shm_masks
    else:
        assert dataset.images is not None and dataset.masks is not None
        dataset_imgs_np = dataset.images
        dataset_masks_np = dataset.masks
    overall_start = 0
    overall_end = len(dataset_imgs_np)
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    start = overall_start + worker_id * per_worker
    end = min(start + per_worker, overall_end)
    dataset.images = list(dataset_imgs_np[start:end])
    dataset.masks = list(dataset_masks_np[start:end])


class HsiDataloaderCutter(torch.utils.data.IterableDataset):
    def __init__(
            self, 
            images, masks,
            augmentation: Optional[Union[DataAugmentator, Callable]] = BaseDataAugmentor(),
            shuffle_data=False,
            cut_window=(8, 8),
            map_mask_to_class=False,
            shared_memory_imgs_data: ShmData = None,
            shared_memory_masks_data: ShmData = None
        ):
        super().__init__()
        self.shuffle_data = shuffle_data
        self.augmentation = augmentation
        if cut_window is not None and cut_window[0] == 1 and cut_window[1] == 1:
            self.ignore_image_augs = True
        else:
            self.ignore_image_augs = False
        self.cut_window = cut_window
        self.map_mask_to_class = map_mask_to_class
        self.shared_memory_imgs_data = shared_memory_imgs_data
        self.shared_memory_masks_data = shared_memory_masks_data
        
        self.shm_imgs: shared_memory.SharedMemory = None
        self.shm_masks: shared_memory.SharedMemory = None

        self.images = images
        self.masks = masks

    def __iter__(self):
        assert self.images is not None and self.masks is not None
        if self.shuffle_data:
            self.images, self.masks = shuffle(self.images, self.masks)
        
        for image, mask in zip(self.images, self.masks):
            yield self.augmentation(
                image, mask, 
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
        image: np.ndarray, h_parts: int, 
        w_parts: int, h_win: int, w_win: int):
    image_parts_list = []

    for h_i in range(h_parts):
        for w_i in range(w_parts):
            img_part = image[:, :,  
                h_i * h_win: (h_i+1) * h_win, 
                w_i * w_win: (w_i+1) * w_win
            ]

            image_parts_list.append(img_part)
    return image_parts_list


def merge_parts_into_single_mask(
        preds, shape, h_parts: int, 
        w_parts: int, h_win: int, w_win: int):
    pred_mask = torch.zeros(
        shape,
        dtype=preds.dtype, device=preds.device
    )
    counter = 0

    for h_i in range(h_parts):
        for w_i in range(w_parts):
            pred_mask[:, :,  
                h_i * h_win: (h_i+1) * h_win, 
                w_i * w_win: (w_i+1) * w_win
            ] = preds[counter]
            counter += 1
    return pred_mask


def collect_prediction_and_target(eval_loader, model, cut_window=(8, 8), image_shape=(512, 512), num_classes=17):
    target_list = []
    pred_list = []
    
    for in_data_x, val_data in iter(eval_loader):
        batch_size = in_data_x.shape[0]
        # We will cut image into peases and stack it into single BIG batch
        h_win, w_win = cut_window
        h_parts, w_parts = image_shape[1] // w_win, image_shape[0] // h_win
        in_data_x_parts_list = cut_into_parts_model_input(
            in_data_x, h_parts=h_parts, 
            w_parts=w_parts, h_win=h_win, w_win=w_win
        )
        in_data_x_batch = torch.cat(in_data_x_parts_list, dim=0) # (N, 17, 1, 1)
        # Make predictions
        preds = model(in_data_x_batch) # (N, num_classes, 8, 8)
        # Create full image again from peases
        pred_mask = merge_parts_into_single_mask(
            preds=preds, shape=(batch_size, num_classes, image_shape[0], image_shape[1]), 
            h_parts=h_parts, w_parts=w_parts, h_win=h_win, w_win=w_win
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


def mask2class(mask):
    # Calculate which class have more pixel count
    max_value = -1
    pixel_count = -1
    for class_indx in np.unique(mask):
        pix_count_s = np.sum(mask == class_indx)
        if pix_count_s > pixel_count:
            max_value = class_indx
            pixel_count = pix_count_s
    assert max_value != -1
    return np.array([max_value], dtype=np.int64) 


def preprocessing(imgs, masks, map_mask_to_class=False, split_size=256):
    with open(f'{PREFIX_INFO_PATH}/data_standartization_params_kfold0.json', 'r') as f:
        data_standartization_params = json.load(f)
    mean = data_standartization_params.get('means')
    std = data_standartization_params.get('stds')
    assert mean is not None and std is not None
    print('Create np array of imgs and masks...')
    imgs_np = np.asarray(imgs, dtype=np.float32) # (N, 237, 1, 1)
    masks_np = np.asarray(masks, dtype=np.int64) # (N, 1, 1, 3)
    print("Split imgs dataset...")
    imgs_split_np = np.array_split(imgs_np, split_size) # (split_size, Ns, 237, 1, 1)
    print('Start preprocess images...')
    # Wo PCA
    # _images = [np.transpose(image, (1, 2, 0)) for image in imgs]
    # W Pca
    with Pool(18) as p:
        _images = list(tqdm(p.imap(
                pca_transformation, 
                imgs_split_np,
                #chunksize=1
            ), total=len(imgs_split_np))
        )
        _images = list(tqdm(p.imap(
            standartization_pool(mean=mean, std=std), 
            _images,
            #chunksize=1
            ), total=len(imgs_split_np))
        )
    _images = list(np.concatenate(_images, axis=0)) # (split_size, Ns, 237, 1, 1) -> (split_size * Ns, 237, 1, 1)
    print("Preprocess masks...")
    _masks = list(np.transpose(masks_np[..., 0:1], (0, -1, 1, 2)))
    print("Finish preprocess!")
    if map_mask_to_class:
        _masks = [mask2class(mask) for mask in _masks]
    return _images, _masks


def test_augmentation(image, mask, **kwargs):
    image = torch.from_numpy(image)
    #image = (image - image.min()) / (image.max() - image.min())
    
    mask = torch.from_numpy(mask)
    mask = torch.squeeze(mask, 0)
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
    aug_random_erase
]


def augmentation(image, mask, map_mask_to_class=False, ignore_image_augs=False):
    image = torch.from_numpy(image)
    mask = torch.from_numpy(mask)
    if not ignore_image_augs:
        for aug_func in AUGS_LIST:
            image, mask = aug_func(image, mask, map_mask_to_class=map_mask_to_class)
    
    #image = (image - image.min()) / (image.max() - image.min())
    mask = torch.squeeze(mask, 0)
    return image, mask


class MySuperNetLittleInput(nn.Module):
    
    def __init__(self, in_f=237, out_f=17, *args):
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

        self.conv5 = nn.Conv2d(64, out_f, kernel_size=3, stride=1, padding=1)
        # (N, 17, 8, 8)
        self.bn5 = nn.BatchNorm2d(out_f)
        self.act5 = nn.ReLU()

        self.final_conv = nn.Conv2d(out_f, out_f, kernel_size=1, stride=1, padding=0)
    
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
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        x = self.final_conv(x)
        return x

    