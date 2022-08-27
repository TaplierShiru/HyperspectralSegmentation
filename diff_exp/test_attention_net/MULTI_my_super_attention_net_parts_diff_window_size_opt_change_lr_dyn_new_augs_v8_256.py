#!/usr/bin/env python
# coding: utf-8
# %%

import os

GPU_ID = "1,2,3"
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
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap
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


def collect_prediction_and_target(eval_loader, model, cut_window=(8, 8), image_shape=(512, 512), num_classes=17, divided_batch=2):
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
    
    for preds, target in zip(pred_list, target_list):
        # preds - (num_classes, H, W)
        preds = preds.detach()
        # target - (H, W)
        target = target.detach()

        preds = nn.functional.softmax(preds, dim=0)
        preds = torch.argmax(preds, dim=0)
        
        preds_one_hoted = torch.nn.functional.one_hot(preds, num_classes).view(-1, num_classes).cpu()
        target_one_hoted = torch.nn.functional.one_hot(target, num_classes).view(-1, num_classes).cpu()
        res = jaccard_score(target_one_hoted, preds_one_hoted, average=None, zero_division=1)
        res_list.append(
            res
        )
    
    res_np = np.stack(res_list)
    #res_np = res_np.mean(axis=0)
    return res_np


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


def take_pred_masks(preds):
    preds = nn.functional.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    return preds



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
    

    def configure_sharded_model(self):
        backbone = auto_wrap(self.model.backbone)
        cam_module = auto_wrap(self.model.cam_module)
        pam_module = auto_wrap(self.model.pam_module)
        final_backbone = auto_wrap(checkpoint_wrapper(self.model.final_backbone))
        
        self.model.backbone = backbone
        self.model.cam_module = cam_module
        self.model.pam_module = pam_module
        self.model.final_backbone =  final_backbone

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
            self.model.parameters(), lr=self.lr
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
        
        def create_big_image(tensor, single_elem_shape):
            shape = tensor.shape
            tensor = torch.cat([
                t_s.view(*single_elem_shape)
                for t_s in tensor
            ], dim=-1)
            return tensor

        pred_tensor, target_tensor = collect_prediction_and_target(outputs, self.model, cut_window=self.cut_window)
        pred_as_mask = take_pred_masks(pred_tensor)
        pred_big_tensor = create_big_image(torch.clone(pred_tensor), single_elem_shape=[1, 17, 512, 512])
        target_big_tensor = create_big_image(torch.clone(target_tensor), single_elem_shape=[1, 512, 512])

        target_one_hotted_tensor = torch.nn.functional.one_hot(
            target_big_tensor, 17 # Num classes
        )
        # (N, H, W, C) --> (N, C, H, W)
        target_one_hotted_tensor = target_one_hotted_tensor.permute(0, -1, 1, 2)
        dice_loss_val = dice_loss(
            pred_big_tensor, target_one_hotted_tensor, 
            dim=[0, 2, 3], use_softmax=True, softmax_dim=1,
        )
        metric = calculate_iou(pred_big_tensor, target_big_tensor)
        
        for batch_idx, (target_s, pred_s) in enumerate(zip(target_tensor, pred_as_mask)):
            if self.enable_image_logging:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                sns.heatmap(pred_s.cpu().detach().numpy(), ax=ax1, vmin=0, vmax=17)
                sns.heatmap(target_s.cpu().detach().numpy(), ax=ax2, vmin=0, vmax=17)
                fig.savefig(f'temp_fig_{GPU_ID}.png')
                plt.close(fig)

                if self.experiment is not None and self.current_epoch != 0:
                    # For Comet logger
                    self.experiment.log_image(
                        f'temp_fig_{GPU_ID}.png', name=f'{batch_idx}', 
                        overwrite=False, step=self.global_step
                    )

        if self.experiment is not None and self.current_epoch != 0:
            # Add confuse matrix
            self.experiment.log_confusion_matrix(
                target_tensor.cpu().detach().numpy().reshape(-1), 
                torch.stack(
                    [elem.cpu() for elem in pred_as_mask], 
                    dim=0
                ).cpu().detach().numpy().reshape(-1)
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
            for i, iou in enumerate(metric.mean(axis=0))
        }
        mean_iou_dict = {
            "mean_iou": float(metric.mean()),
        }
        
        # Log this metric in order to save checkpoint of experements
        self.log_dict(mean_iou_dict)
        
        if self.experiment is not None and self.current_epoch != 0:
        
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
    image = (image - image.min()) / (image.max() - image.min())
    
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
    image = (image - image.min()) / (image.max() - image.min())

    mask = torch.from_numpy(mask)
    if not ignore_image_augs:
        for aug_func in AUGS_LIST:
            image, mask = aug_func(image, mask, map_mask_to_class=map_mask_to_class)
    
    mask = torch.squeeze(mask, 0)
    return image, mask


# Paper must read:
# https://arxiv.org/pdf/1904.11492.pdf
#
# Original paper:
# https://arxiv.org/pdf/1906.02849.pdf
#
# Github:
# https://github.com/sinAshish/Multi-Scale-Attention
#


class PAM_Module(nn.Module):
    """ 
    Position attention module

    """
    #Ref from SAGAN
    def __init__(self, in_dim, dim_reduse: int = 8):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // dim_reduse, 
            kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // dim_reduse,
             kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, 
            kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_attention=False):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out

        if return_attention:
            return out, attention

        return out


class CAM_Module(nn.Module):
    """ 
    Channel attention module

    """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, return_attention=False):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
       
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy # swap?
        attention = self.softmax(energy_new) # Sigmoid?
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out

        if return_attention:
            return out, attention

        return out



class MySuperNetLittleInput(nn.Module):
    
    def __init__(self, in_f=237, out_f=17, *args):
        super().__init__()
        #self.bn_start = nn.BatchNorm3d(in_f)

        self.backbone = nn.Sequential(
            nn.Conv2d(in_f, 96, kernel_size=3, stride=1, padding=1, bias=False),
            # (N, 128, 8, 8)
            nn.LayerNorm([96, 256, 256]),
            nn.ReLU(),
        
            nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1, bias=False),
            # (N, 64, 8, 8)
            nn.LayerNorm([48, 256, 256]),
            nn.ReLU(),
            
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
            # (N, 32, 8, 8)
            nn.LayerNorm([48, 256, 256]),
            nn.ReLU(),
            
            nn.Conv2d(48, out_f, kernel_size=3, stride=1, padding=1, bias=False),
            # (N, out_f, 8, 8)
            nn.LayerNorm([out_f, 256, 256]),
            nn.ReLU(),
        )
        # Output (N, 17, 8, 8)
        self.cam_module = CAM_Module(out_f)
        # Output (N, 17, 8, 8)
        self.pam_module = PAM_Module(out_f, dim_reduse=1)
        # Concat out_f and out_f = out_f * 2
        self.final_backbone = nn.Sequential(
            nn.Conv2d(out_f, out_f, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LayerNorm([out_f, 256, 256]),
            nn.ReLU(),
            # Final conv
            nn.Conv2d(out_f, out_f, kernel_size=1, stride=1, padding=0, bias=False),
        )
    
    def forward(self, x):
        x = self.backbone(x)

        cam_x = self.cam_module(x)
        pam_x = self.pam_module(x)

        x = cam_x + pam_x + x
        x = self.final_backbone(x)

        return x


USE_SHM = True

if USE_SHM:
    NUM_WORKERS = 10
else:
    NUM_WORKERS = 5

N_REPEAT_EXP = 2

T_0_param = 2
T_mult_param = 1


CUT_WINDOW_SIZE_LIST = [
#    { "cut_window": (1, 1),     "batch_size": 32768, "lr": 5e-2, "max_epochs": 5   },  # Max: 32768
#    { "cut_window": (2, 2),     "batch_size": 4096,  "lr": 2e-2, "max_epochs": 5   },  # Max: 4096
#    { "cut_window": (4, 4),     "batch_size": 512,   "lr": 4e-3, "max_epochs": 5   },  # Max: 512
#    { "cut_window": (8, 8),     "batch_size": 128,   "lr": 1e-3, "max_epochs": 5   },  # Max: 128
#    { "cut_window": (16, 16),   "batch_size": 128,   "lr": 1e-3, "max_epochs": 5   },  # Max: 128
#    { "cut_window": (32, 32),   "batch_size": 128,   "lr": 1e-3, "max_epochs": 7   },  # Max: 128
#    { "cut_window": (64, 64),   "batch_size": 128,   "lr": 1e-3, "max_epochs": 10  },  # Max: 128
#    { 
#        "cut_window": (128, 128), "batch_size": 4, "lr": 1e-3, 
#        "max_epochs": 400,  "epoch_list": None, "lr_list": None,
#        #             epoch_list=[-1, 100, 200]   lr_list=[1e-3, 1e-4, 1e-5]
#    },  # Max: 128
    { 
        "cut_window": (256, 256), "batch_size": 1, "lr": 1e-3, 
        "max_epochs": 440,  "epoch_list": None, "lr_list": None,
        #             epoch_list=[-1, 100, 200]   lr_list=[1e-3, 1e-4, 1e-5]
    },  # Max: 64
#    { 
#        "cut_window": (512, 512), "batch_size": 8, "lr": 5e-4, 
#        "max_epochs": 500,  "epoch_list": None, "lr_list": None,
#        #             epoch_list=[-1, 100, 200]   lr_list=[1e-3, 1e-4, 1e-5]
#    },  # Max: 22
]


NAME2CLASS = {
    MySuperNetLittleInput.__name__:             MySuperNetLittleInput,
}

ARCH_TYPES = [
    (MySuperNetLittleInput.__name__,            {}),
]


def return_net(arch_type: str, in_f: int, out_f: int):
    class_net = NAME2CLASS.get(arch_type)
    if class_net is None:
        raise TypeError(f"Unknown type for arch, type={arch_type}")

    if arch_type == 'Unet':
        net = class_net(in_channels=in_f, out_channels=out_f, init_features=128, pretrained=False)
    else:
        net = class_net(in_f, out_f)
    return net



def start_exp(
        number_of_exp: int, T_0: int, T_mult: int, 
        arch_type: str, batch_size: int, cut_window: tuple,
        lr: float, max_epochs: int, lr_list: list = None, epoch_list: list = None):

    print(f'=======START EXP NUMBER {number_of_exp} ==============')
    print(f'Params: number_of_exp={number_of_exp}, T_0={T_0} \n' +\
          f'T_mult={T_mult}, arch_type={arch_type}, batch_size={batch_size} \n' +\
          f'cut_window={cut_window}, lr={lr}, \n' +\
          f'lr_list={lr_list}, epoch_list={epoch_list}'
    )
    test_indices = np.load(f'{PREFIX_INFO_PATH}/kfold0_indx_test.npy')
    train_indices = np.load(f'{PREFIX_INFO_PATH}/kfold0_indx_train.npy')

    # Load and preprocess data
    # Train
    if USE_SHM:
        print("Using shared memory...")
        dataset_creator_train = DatasetCreator(
            PATH_DATA, preprocessing=preprocessing, 
            indices=train_indices, cut_window=cut_window,
            create_shared_memory=True, shuffle_then_prepared=True
        )
        dataset_train = HsiDataloaderCutter(
            images=None, masks=None,
            augmentation=augmentation,
            shuffle_data=True, cut_window=cut_window,
            shared_memory_imgs_data=dataset_creator_train.data_shm_imgs,
            shared_memory_masks_data=dataset_creator_train.data_shm_masks,
        )
    else:
        dataset_creator_train = DatasetCreator(
            PATH_DATA, preprocessing=preprocessing, 
            indices=train_indices, cut_window=cut_window,
            create_shared_memory=False, shuffle_then_prepared=True
        )
        dataset_train = HsiDataloaderCutter(
            images=dataset_creator_train.images, masks=dataset_creator_train.masks,
            augmentation=augmentation,
            shuffle_data=True, cut_window=cut_window,
        )
    print(f"Number of workers={NUM_WORKERS}")
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, 
        num_workers=NUM_WORKERS, pin_memory=False, prefetch_factor=2,
        worker_init_fn=worker_init_fn, drop_last=True
    )
    # Test
    dataset_creator_test = DatasetCreator(
        PATH_DATA, preprocessing=preprocessing, 
        indices=test_indices, cut_window=None,
        create_shared_memory=False
    )
    dataset_test = HsiDataloaderCutter(
        images=dataset_creator_test.images, masks=dataset_creator_test.masks,
        augmentation=test_augmentation,
        shuffle_data=False, cut_window=None,
    )
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1)
    
    net = return_net(arch_type=arch_type, in_f=17, out_f=17)
    net.eval()
    with torch.no_grad():
        _ = net(torch.randn(1, 17, cut_window[0], cut_window[1]))
    net.train()
    comet_exp = comet_ml.Experiment(
        api_key="your-key",
        workspace="your-workspace",  # Optional
        project_name="your-project-name",  # Optional
    )
    name_exp = f"Attention moduleV8 test cut_window={cut_window} (run {number_of_exp})" +\
               f"change_lr_dyn rerun=1 | {max_epochs}ep | PCA. | RustamPreprocess(k=1) | CE " +\
               f"cosine(t_0={T_0},t_mul={T_mult}) lr={lr} | arch_type={arch_type} | batch_size={batch_size}"
    comet_exp.set_name(name_exp)
    comet_exp.add_tag('attention_module_test')

    model = NnModel(net, nn.CrossEntropyLoss(), experiment=comet_exp,
        T_0=T_0, T_mult=T_mult, cut_window=cut_window, lr=lr,
        epoch_list=epoch_list, lr_list=lr_list,
    )

    # saves a checkpoint-file
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"pytorch_li_logs/rerun=8(v8)/(run={number_of_exp})Attention_module_test_"+\
                f"cut_window={cut_window}_change_lr_dyn_"+\
                f"arch_{max_epochs}ep_PCA_RustamPreprocess(k=1)_CE_"+\
                f"cosine(t_0={T_0},t_mul={T_mult})_arch_type={arch_type}_" +\
                f"batch_size={batch_size}_lr={lr}",
        monitor="mean_iou",
        filename="model-{epoch:02d}-{mean_iou:.2f}",
        save_top_k=-1,
        mode="min",
    )


    trainer = pl.Trainer(
        gpus=3, strategy="deepspeed_stage_2", 
        max_epochs=max_epochs,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, test_loader)

    comet_exp.end()
    # Make sure shared memory is close and unlinked
    # In order to clean memory
    if USE_SHM:
        print('Clear shared memory...')
        dataset_creator_train.close_shm()

    print(f'=======END EXP NUMBER {number_of_exp} ================')


def main():
    for arch_type, params in ARCH_TYPES:
        for window_params in CUT_WINDOW_SIZE_LIST:
            params.update(window_params)
            for number_of_exp in range(1, N_REPEAT_EXP+1):
                start_exp(
                    number_of_exp=number_of_exp, T_0=T_0_param,
                    T_mult=T_mult_param, arch_type=arch_type, **params
                )


if __name__ == '__main__':
    main()

