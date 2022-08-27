#!/usr/bin/env python
# coding: utf-8
# %%

import numpy as np
import os
import sys

np.set_printoptions(suppress=True)
GPU_ID = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

sys.path.append('/home/rustam/hyperspecter_segmentation/makitorch')
sys.path.append('/home/rustam/hyperspecter_segmentation/')


import random
from multiprocessing.dummy import Pool
from multiprocessing import shared_memory
import copy
from makitorch import *
from makitorch.dataset_remapper import DatasetRemapper
import math
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
import glob
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


#
# ----------------------------------------------------------------
#                    CONTANTS
# ----------------------------------------------------------------
#

DATA_SHAPE = (128, 128)
INPUT_DIM = 17
NUM_CLASSES = 7
NUMBER_RESULTS_TO_PLOT = 32


KFOLD = 'kfold1'
PREFIX_INFO_PATH = f'/home/rustam/hyperspecter_segmentation/danil_cave/kfolds_data_with_other/{KFOLD}/data'
PATH_DATA = f'/raid/rustam/hyperspectral_dataset/diff_exp_with_other__attention_with_other/{KFOLD}'

PREFIX_TRAIN = 'train'
PREFIX_TEST = 'test'
PREFIX_TEST_UNKNOWN = 'test_unknown'
PREFIX_HSI = 'hsi'
PREFIX_MASKS = 'masks'

PATH_TRAIN_DATA = f'{PATH_DATA}/{PREFIX_TRAIN}'
PATH_TEST_DATA = f'{PATH_DATA}/{PREFIX_TEST}'
PATH_TEST_UNKNOWN_DATA = f'{PATH_DATA}/{PREFIX_TEST_UNKNOWN}'

PATH_TRAIN_HSI_DATA = f'{PATH_TRAIN_DATA}/{PREFIX_HSI}'
PATH_TRAIN_MASKS_DATA = f'{PATH_TRAIN_DATA}/{PREFIX_MASKS}'
PATH_TEST_HSI_DATA = f'{PATH_TEST_DATA}/{PREFIX_HSI}'
PATH_TEST_MASKS_DATA = f'{PATH_TEST_DATA}/{PREFIX_MASKS}'
PATH_TEST_UNKNOWN_HSI_DATA = f'{PATH_TEST_UNKNOWN_DATA}/{PREFIX_HSI}'
PATH_TEST_UNKNOWN_MASKS_DATA = f'{PATH_TEST_UNKNOWN_DATA}/{PREFIX_MASKS}'

device = 'cuda:0'
pca_explained_variance = np.load(f'{PREFIX_INFO_PATH}/{KFOLD}_PcaExplainedVariance_.npy')
pca_mean = np.load(f'{PREFIX_INFO_PATH}/{KFOLD}_PcaMean.npy')
pca_components = np.load(f'{PREFIX_INFO_PATH}/{KFOLD}_PcaComponents.npy')
DATA_STANDARTIZATION_PARAMS_PATH = f'{PREFIX_INFO_PATH}/data_standartization_params_{KFOLD}.json'
with open(DATA_STANDARTIZATION_PARAMS_PATH, 'r') as f:
    DATA_STANDARTIZATION_PARAMS = json.load(f)


USE_SHM = True

if USE_SHM:
    NUM_WORKERS = 10
else:
    NUM_WORKERS = 5

N_REPEAT_EXP = 4

T_0_param = 2
T_mult_param = 1


EXP_DATA_LIST = [
#    { batch_size": 32768, "lr": 5e-2, "max_epochs": 5   },  # Max: 32768
#    { batch_size": 4096,  "lr": 2e-2, "max_epochs": 5   },  # Max: 4096
#    { batch_size": 512,   "lr": 4e-3, "max_epochs": 5   },  # Max: 512
#    { batch_size": 128,   "lr": 1e-3, "max_epochs": 5   },  # Max: 128
#    { batch_size": 128,   "lr": 1e-3, "max_epochs": 5   },  # Max: 128
#    { batch_size": 128,   "lr": 1e-3, "max_epochs": 7   },  # Max: 128
#    { batch_size": 128,   "lr": 1e-3, "max_epochs": 10  },  # Max: 128
    { 
        "batch_size": 48, "lr": 1e-3, 
        "max_epochs": 200,  "epoch_list": None, "lr_list": None,
        #             epoch_list=[-1, 100, 200]   lr_list=[1e-3, 1e-4, 1e-5]
    },  # Max: 128
#    { 
#        "batch_size": 16, "lr": 5e-4, 
#        "max_epochs": 400,  "epoch_list": None, "lr_list": None,
#        #             epoch_list=[-1, 100, 200]   lr_list=[1e-3, 1e-4, 1e-5]
#    },  # Max: 64
#    { 
#        "batch_size": 8, "lr": 5e-4, 
#        "max_epochs": 500,  "epoch_list": None, "lr_list": None,
#        #             epoch_list=[-1, 100, 200]   lr_list=[1e-3, 1e-4, 1e-5]
#    },  # Max: 22
]

#
# ----------------------------------------------------------------
#                    DATASET PIPELINE
# ----------------------------------------------------------------
#

class ShmData:

    def __init__(self, shm_name, shape, dtype):
        self.shm_name = shm_name
        self.shape = shape
        self.dtype = dtype


class DatasetCreator:

    def __init__(
            self, 
            path_hsi: str,
            path_masks: str,
            preprocessing: Optional[Union[DataPreprocessor, Callable]] = BaseDataPreprocessor(),
            create_shared_memory=False,
            shuffle_then_prepared=False,
            path_old2new: str = None):
        self.preprocessing = preprocessing
        self.create_shared_memory = create_shared_memory
        
        if path_old2new:
            self.old2new_mapper = DatasetRemapper(np.load(path_old2new))
        else:
            self.old2new_mapper = None

        self.hsis = []
        self.masks = []
        self._shm_hsis = None
        self._shm_masks = None
        self.data_shm_hsis: Optional[ShmData] = None
        self.data_shm_masks: Optional[ShmData] = None

        path_hsi_list = glob.glob(os.path.join(path_hsi, '*.npy'))
        path_masks_list = glob.glob(os.path.join(path_masks, '*.npy'))

        for path_hsi_s, path_mask_s in tqdm(zip(path_hsi_list, path_masks_list)):
            hsi, mask = np.load(path_hsi_s), np.load(path_mask_s)

            if self.old2new_mapper:
                _, mask = self.old2new_mapper(None, mask)
            self.hsis.append(hsi)
            self.masks.append(mask)

        if self.preprocessing is not None:
            print("Preprocess data...")
            self.hsis, self.masks = self.preprocessing(
                self.hsis, self.masks
            )

        if shuffle_then_prepared:
            self.hsis, self.masks = shuffle(self.hsis, self.masks)

        # Create shared memory
        if create_shared_memory:
            print('Create shared memory...')
            # First - map hsis and masks into np
            self.hsis = np.asarray(self.hsis, dtype=np.float32)
            self.masks = np.asarray(self.masks, dtype=np.int64)
            # Imgs
            shm_hsis = shared_memory.SharedMemory(create=True, size=self.hsis.nbytes)
            shm_hsis_arr = np.ndarray(self.hsis.shape, dtype=self.hsis.dtype, buffer=shm_hsis.buf)
            shm_hsis_arr[:] = self.hsis[:]
            self.hsis = shm_hsis_arr # Do not keep dublicate 
            self.data_shm_hsis = ShmData(
                shm_name=shm_hsis.name, shape=self.hsis.shape, 
                dtype=self.hsis.dtype
            )
            self._shm_hsis = shm_hsis
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
            print("Shared memory are created for hsis and masks!")

    def close_shm(self):
        if self.create_shared_memory:
            # Make sure there is no reference data to hsis/masks
            del self.hsis
            del self.masks
            self.hsis = []
            self.masks = []
            # Close and unlink
            if self._shm_masks is not None:
                self._shm_masks.close()
                self._shm_masks.unlink()
                self._shm_masks = None

            if self._shm_hsis is not None:
                self._shm_hsis.close()
                self._shm_hsis.unlink()
                self._shm_hsis = None
            print("Shared memory for masks and hsis are success cleared!")


# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    shared_memory_hsis_data: ShmData = dataset.shared_memory_hsis_data
    shared_memory_masks_data: ShmData = dataset.shared_memory_masks_data
    if shared_memory_hsis_data is not None and shared_memory_masks_data is not None:
        # Take array from memory
        existing_shm_hsis = shared_memory.SharedMemory(name=shared_memory_hsis_data.shm_name)
        dataset_hsis_np = np.ndarray(
            shared_memory_hsis_data.shape, 
            dtype=shared_memory_hsis_data.dtype, buffer=existing_shm_hsis.buf
        )
        dataset.shm_hsis = existing_shm_hsis
        existing_shm_masks = shared_memory.SharedMemory(name=shared_memory_masks_data.shm_name)
        dataset_masks_np = np.ndarray(
            shared_memory_masks_data.shape, 
            dtype=shared_memory_masks_data.dtype, buffer=existing_shm_masks.buf
        )
        dataset.shm_masks = existing_shm_masks
    else:
        assert dataset.hsis is not None and dataset.masks is not None
        dataset_hsis_np = dataset.hsis
        dataset_masks_np = dataset.masks
    overall_start = 0
    overall_end = len(dataset_hsis_np)
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    start = overall_start + worker_id * per_worker
    end = min(start + per_worker, overall_end)
    dataset.hsis = list(dataset_hsis_np[start:end])
    dataset.masks = list(dataset_masks_np[start:end])


class HsiDatasetDataloader(torch.utils.data.IterableDataset):
    def __init__(
            self, 
            hsis, masks,
            augmentation: Optional[Union[DataAugmentator, Callable]] = BaseDataAugmentor(),
            shuffle_data=False,
            shared_memory_hsis_data: ShmData = None,
            shared_memory_masks_data: ShmData = None
        ):
        super().__init__()
        self.shuffle_data = shuffle_data
        self.augmentation = augmentation

        self.shared_memory_hsis_data = shared_memory_hsis_data
        self.shared_memory_masks_data = shared_memory_masks_data
        
        self.shm_hsis: shared_memory.SharedMemory = None
        self.shm_masks: shared_memory.SharedMemory = None

        self.hsis = hsis
        self.masks = masks

    def __iter__(self):
        assert self.hsis is not None and self.masks is not None, 'Dataloader is not prepared'
        if self.shuffle_data:
            self.hsis, self.masks = shuffle(self.hsis, self.masks)
        
        for image, mask in zip(self.hsis, self.masks):
            # Make sure to copy them!
            yield self.augmentation(
                image.copy(), mask.copy(),
            )


#
# ----------------------------------------------------------------
#                    MODEL TOOLS
# ----------------------------------------------------------------
#

#   TODO: Check this line, previously were situations when some images are skipped
def collect_prediction_and_target(eval_loader, model, divided_batch=2):
    target_list = []
    pred_list = []
    
    for in_data_x_batch, val_data in iter(eval_loader):
        batch_size = in_data_x_batch.shape[0]
        # Make predictions | TODO: Here will be skiped - remaining from divide does not plus
        part_divided = (batch_size // divided_batch) + (batch_size % divided_batch != 0)
        pred_batch_list = []
        for b_i in range(part_divided):
            if b_i == (part_divided - 1):
                # last
                single_batch = in_data_x_batch[b_i * divided_batch:]
            else:
                single_batch = in_data_x_batch[b_i * divided_batch: (b_i+1) * divided_batch]
            # Make predictions
            preds = model(single_batch) # (divided_batch, num_classes, H, W)
            pred_batch_list.append(preds)
        pred_mask = torch.cat(pred_batch_list, dim=0)  # (N, num_classes, H, W)

        target_list.append(val_data)
        pred_list.append(pred_mask)
    return (torch.cat(pred_list, dim=0), 
            torch.cat(target_list, dim=0)
    )
        
        

def calculate_iou(pred_list, target_list):
    res_list = []
    
    for preds, target in zip(pred_list, target_list):
        # preds - (num_classes, H, W)
        preds = preds.detach()
        # target - (H, W)
        target = target.detach()

        preds = nn.functional.softmax(preds, dim=0)
        preds = torch.argmax(preds, dim=0)
        
        preds_one_hoted = torch.nn.functional.one_hot(preds, NUM_CLASSES).view(-1, NUM_CLASSES).cpu()
        target_one_hoted = torch.nn.functional.one_hot(target, NUM_CLASSES).view(-1, NUM_CLASSES).cpu()
        res = jaccard_score(target_one_hoted, preds_one_hoted, average=None, zero_division=1)
        res_list.append(
            res
        )

    res_np = np.stack(res_list)
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


# Some write that its better to init network with this
# But When I try it - it does not differ from default Init
# So, just keep it here for a while...
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if getattr(m, 'bias') is not None:
            m.bias.data.fill_(0.01)


#
# ----------------------------------------------------------------
#                  PYTORCH LIGHTNING MODEL
# ----------------------------------------------------------------
#

class NnModel(pl.LightningModule):
    def __init__(
            self, model, loss,
            T_0=10, T_mult=2, experiment=None, enable_image_logging=True,
            lr=1e-3, lr_list=None, epoch_list=None):
        super().__init__()
        self.model = model
        self.loss = loss
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
            # Make copy to make sure that original data is safe
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
        # Stuff here - for change learning rate on a fly
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
        # Change lr after some epoch, original idea from:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3095
        # But here I do something different and its more what I like
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
        preds = self.model(img) # (N, C, H, W)
        loss = self.loss(preds, mask) # (N, H, W)
        self.log('train_loss', loss)
        if self.experiment is not None:
            self.experiment.log_metric("train_loss", loss, epoch=self.current_epoch, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        return batch
    
    def validation_epoch_end(self, outputs):
        print('VALIDATION: Size epoch end input with data n=', len(outputs))
        
        def create_big_image(tensor, single_elem_shape):
            shape = tensor.shape
            tensor = torch.cat([
                t_s.view(*single_elem_shape)
                for t_s in tensor
            ], dim=-1)
            return tensor
        # (batch_size, NUM_CLASSES, H, W) | (batch_size, H, W)
        pred_tensor, target_tensor = collect_prediction_and_target(outputs, self.model)
        pred_as_mask = take_pred_masks(pred_tensor) # (batch_size, H, W)
        pred_big_tensor = create_big_image(
            pred_tensor, 
            single_elem_shape=[1, NUM_CLASSES, DATA_SHAPE[0], DATA_SHAPE[1]]
        )
        target_big_tensor = create_big_image(
            target_tensor, 
            single_elem_shape=[1, DATA_SHAPE[0], DATA_SHAPE[1]]
        )

        target_one_hotted_tensor = torch.nn.functional.one_hot(
            target_big_tensor, NUM_CLASSES # Num classes
        )
        # (N, H, W, C) --> (N, C, H, W)
        target_one_hotted_tensor = target_one_hotted_tensor.permute(0, -1, 1, 2)
        dice_loss_val = dice_loss(
            pred_big_tensor, target_one_hotted_tensor, 
            dim=[0, 2, 3], use_softmax=True, softmax_dim=1,
        )
        metric = calculate_iou(pred_big_tensor, target_big_tensor)
        
        if self.enable_image_logging:
            for batch_indx, (target_s, pred_s) in enumerate(zip(target_tensor, pred_as_mask)):
                target_s = target_s.squeeze()
                pred_s = pred_s.squeeze()
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                sns.heatmap(pred_s.cpu().detach().numpy(), ax=ax1, vmin=0, vmax=NUM_CLASSES)
                sns.heatmap(target_s.cpu().detach().numpy(), ax=ax2, vmin=0, vmax=NUM_CLASSES)
                fig.savefig(f'temp_fig_{GPU_ID}-wo.png')
                plt.close(fig)

                if self.experiment is not None and self.current_epoch != 0:
                    self.experiment.log_image(
                        f'temp_fig_{GPU_ID}-wo.png', name=f'{batch_indx}', 
                        overwrite=False, step=self.global_step
                    )
                # Sometimes - send too much images to comet - slow training
                # So, for training time skip some samples - its better to do this after 
                # training on the best model
                if batch_indx == NUMBER_RESULTS_TO_PLOT:
                    break

        if self.experiment is not None and self.current_epoch != 0:
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
        
        # Log will save stuff in comet
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

    def test_step(self, batch, batch_idx):
        return batch

    def test_epoch_end(self, outputs):
        # This function almost same as validation_epoch_end - but its only for test data
        # TODO: Make it as one function?
        print('TEST: Performe test with data n=', len(outputs))
        
        def create_big_image(tensor, single_elem_shape):
            shape = tensor.shape
            tensor = torch.cat([
                t_s.view(*single_elem_shape)
                for t_s in tensor
            ], dim=-1)
            return tensor
        # (batch_size, NUM_CLASSES, H, W) | (batch_size, H, W)
        pred_tensor, target_tensor = collect_prediction_and_target(outputs, self)
        pred_as_mask = take_pred_masks(pred_tensor) # (batch_size, H, W)
        pred_big_tensor = create_big_image(
            pred_tensor, 
            single_elem_shape=[1, NUM_CLASSES, DATA_SHAPE[0], DATA_SHAPE[1]]
        )
        target_big_tensor = create_big_image(
            target_tensor, 
            single_elem_shape=[1, DATA_SHAPE[0], DATA_SHAPE[1]]
        )

        target_one_hotted_tensor = torch.nn.functional.one_hot(
            target_big_tensor, NUM_CLASSES # Num classes
        )
        # (N, H, W, C) --> (N, C, H, W)
        target_one_hotted_tensor = target_one_hotted_tensor.permute(0, -1, 1, 2)
        dice_loss_val = dice_loss(
            pred_big_tensor, target_one_hotted_tensor, 
            dim=[0, 2, 3], use_softmax=True, softmax_dim=1,
        )
        metric = calculate_iou(pred_big_tensor, target_big_tensor)
        
        if self.enable_image_logging:
            for batch_indx, (target_s, pred_s) in enumerate(zip(target_tensor, pred_as_mask)):
                target_s = target_s.squeeze()
                pred_s = pred_s.squeeze()
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                sns.heatmap(pred_s.cpu().detach().numpy(), ax=ax1, vmin=0, vmax=NUM_CLASSES)
                sns.heatmap(target_s.cpu().detach().numpy(), ax=ax2, vmin=0, vmax=NUM_CLASSES)
                fig.savefig(f'test_unknown_temp_fig_{GPU_ID}-wo.png')
                plt.close(fig)

                if self.experiment is not None and self.current_epoch != 0:
                    self.experiment.log_image(
                        f'test_unknown_temp_fig_{GPU_ID}-wo.png', 
                        name=f'unknown_{batch_indx}', 
                        overwrite=False, step=self.global_step
                    )

        if self.experiment is not None and self.current_epoch != 0:
            self.experiment.log_confusion_matrix(
                target_tensor.cpu().detach().numpy().reshape(-1), 
                torch.stack(
                    [elem.cpu() for elem in pred_as_mask], 
                    dim=0
                ).cpu().detach().numpy().reshape(-1)
            )

        mean_dice_loss_per_class_dict = {
            f"test_mean_dice_loss_per_class_{i}": d_l.float()
            for i, d_l in enumerate(dice_loss_val)
        }
        mean_dice_loss_dict = {
            f"test_mean_dice_loss": dice_loss_val.mean().float()
        }
        mean_iou_class_dict = {
            f"test_mean_iou_class_{i}": torch.tensor(iou, dtype=torch.float)
            for i, iou in enumerate(metric.mean(axis=0))
        }
        mean_iou_dict = {
            "test_mean_iou": float(metric.mean()),
        }
        
        # Log will save stuff in comet
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


#
# ----------------------------------------------------------------
#                   PREPROCESS TOOLS
# ----------------------------------------------------------------
#

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


def preprocessing(hsis, masks, split_size=256):
    mean, std = (
        DATA_STANDARTIZATION_PARAMS.get('means'), 
        DATA_STANDARTIZATION_PARAMS.get('stds')
    )
    assert mean is not None and std is not None
    # Masks shape (N, 1, H, W)
    # Hsis shape (N, 237, H, W)
    print('Create np array of hsis and masks...')
    hsis_np = np.asarray(hsis, dtype=np.float32)
    print("Split hsis dataset...")
    hsis_split_np = np.array_split(hsis_np, split_size) # (split_size, Ns, 237, H, W)
    print('Start preprocess hsis...')
    # Wo PCA
    # _hsis = [np.transpose(hsi, (1, 2, 0)) for hsi in hsis]
    # W Pca
    with Pool(18) as p:
        _hsis = list(tqdm(p.imap(
                pca_transformation, 
                hsis_split_np,
                #chunksize=1
            ), total=len(hsis_split_np))
        )
        _hsis = list(tqdm(p.imap(
                standartization_pool(mean=mean, std=std), 
                _hsis,
                #chunksize=1
            ), total=len(_hsis))
        )
    _hsis = list(np.concatenate(_hsis, axis=0)) # (split_size, Ns, 237, 1, 1) -> (split_size * Ns, 237, 1, 1)
    print("Finish preprocess!")
    return _hsis, masks


def test_augmentation(hsi, mask, **kwargs):
    hsi = torch.from_numpy(hsi)
    #image = (image - image.min()) / (image.max() - image.min())
    
    mask = torch.from_numpy(mask) # (1, H, W)
    # Target mask must be with shape (H, W)
    mask = torch.squeeze(mask, 0)
    return hsi, mask

#
# ----------------------------------------------------------------
#                           AUGMENTATION
# ----------------------------------------------------------------
#

def aug_random_rotate(hsi, mask, **kwargs):
    angle = T.RandomRotation.get_params((-30, 30))
    hsi = TF.rotate(hsi, angle, interpolation=T.InterpolationMode.BILINEAR)
    mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
    return hsi, mask


def aug_flip_horizontal(hsi, mask, **kwargs):
    if torch.rand(1) > 0.5:
        hsi = TF.hflip(hsi)
        mask = TF.hflip(mask)
    return hsi, mask


def aug_flip_vertical(hsi, mask, **kwargs):
    if torch.rand(1) > 0.5:
        hsi = TF.vflip(hsi)
        mask = TF.vflip(mask)
    return hsi, mask


MASK_AUG_SCALE = 100
MASK_AUG_COMPARE = 90
RandomEraseTorch = T.RandomErasing(
    p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), 
    value='random', inplace=False
)
def aug_random_erase(hsi, mask, **kwargs):
    # Create mask in order to take area of aug
    mask_aug_area = torch.ones(
        1, hsi.shape[1], hsi.shape[2], 
        dtype=hsi.dtype, device=hsi.device
    ) * MASK_AUG_SCALE
    # Apply aug both - on hsi and mask
    aug_data_input = torch.cat([hsi, mask_aug_area], dim=0)
    # Apply aug
    aug_data_output = RandomEraseTorch(aug_data_input)
    hsi, mask_aug = aug_data_output[:-1], aug_data_output[-1:]  # slice hsi and mask
    # Take mask and reverse values
    # 0 - cutout zone, 1 - good zone
    mask_aug = (mask_aug > MASK_AUG_COMPARE).long()
    mask = mask * mask_aug
    return hsi, mask


AUGS_LIST = [
    aug_random_rotate,
    aug_flip_horizontal,
    aug_flip_vertical,
    aug_random_erase
]


def augmentation(hsi, mask, ignore_image_augs=False):
    hsi = torch.from_numpy(hsi)
    mask = torch.from_numpy(mask)
    if not ignore_image_augs:
        for aug_func in AUGS_LIST:
            hsi, mask = aug_func(hsi, mask)
    
    #image = (image - image.min()) / (image.max() - image.min())
    # Target mask must be with shape (H, W)
    mask = torch.squeeze(mask, 0)
    return hsi, mask


#
# ----------------------------------------------------------------
#                           MODEL
# ----------------------------------------------------------------


class MySuperNetLittleInput(nn.Module):
    
    def __init__(self, in_f=237, out_f=NUM_CLASSES, *args):
        super().__init__()
        #self.bn_start = nn.BatchNorm3d(in_f)

        self.backbone = nn.Sequential(
            # (N, in_f, 128, 128)
            nn.Conv2d(in_f, 64, kernel_size=5, stride=2, padding=2, bias=False),
            # (N, 128, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            # (N, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # (N, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128 * 3, kernel_size=3, stride=1, padding=1, bias=False),
            # (N, 128 * 3, 32, 32)
            nn.BatchNorm2d(128 * 3),
            nn.ReLU(),
        )
        self.final_backbone = nn.Sequential(
            nn.Conv2d(128 * 3, 128, kernel_size=1, stride=1, padding=0, bias=False),
            # (N, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # (N, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            # (N, 128, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            # (N, 64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, out_f, kernel_size=5, stride=1, padding=2, bias=False),
            # (N, out_f, 128, 128)
            nn.BatchNorm2d(out_f),
            nn.ReLU(),
            # Final conv
            nn.Conv2d(out_f, out_f, kernel_size=1, stride=1, padding=0, bias=False),
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.final_backbone(x)
        return x


#
# ----------------------------------------------------------------
#                   START EXPERIMENT TOOLS
# ----------------------------------------------------------------
#

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
        arch_type: str, batch_size: int,
        lr: float, max_epochs: int, lr_list: list = None, epoch_list: list = None):

    print(f'=======START EXP NUMBER {number_of_exp} ==============')
    print(f'Params: number_of_exp={number_of_exp}, T_0={T_0} \n' +\
          f'T_mult={T_mult}, arch_type={arch_type}, batch_size={batch_size} \n' +\
          f'lr={lr}, \n' +\
          f'lr_list={lr_list}, epoch_list={epoch_list}'
    )

    # Load and preprocess data
    # Train
    if USE_SHM:
        print("Using shared memory...")
        dataset_creator_train = DatasetCreator(
            path_hsi=PATH_TRAIN_HSI_DATA,
            path_masks=PATH_TRAIN_MASKS_DATA,
            preprocessing=preprocessing, 
            create_shared_memory=True, shuffle_then_prepared=True
        )
        dataset_train = HsiDatasetDataloader(
            hsis=None, masks=None,
            augmentation=augmentation,
            shuffle_data=True,
            shared_memory_hsis_data=dataset_creator_train.data_shm_hsis,
            shared_memory_masks_data=dataset_creator_train.data_shm_masks,
        )
    else:
        dataset_creator_train = DatasetCreator(
            path_hsi=PATH_TRAIN_HSI_DATA,
            path_masks=PATH_TRAIN_MASKS_DATA,
            preprocessing=preprocessing, 
            create_shared_memory=False, shuffle_then_prepared=True
        )
        dataset_train = HsiDatasetDataloader(
            hsis=dataset_creator_train.hsis, masks=dataset_creator_train.masks,
            augmentation=augmentation,
            shuffle_data=True,
        )
    print(f"Number of workers={NUM_WORKERS}")
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, 
        num_workers=NUM_WORKERS, pin_memory=False, prefetch_factor=2,
        worker_init_fn=worker_init_fn, drop_last=True
    )
    # Validation
    dataset_creator_val = DatasetCreator(
        path_hsi=PATH_TEST_HSI_DATA,
        path_masks=PATH_TEST_MASKS_DATA,
        preprocessing=preprocessing,
        create_shared_memory=False
    )
    dataset_val = HsiDatasetDataloader(
        hsis=dataset_creator_val.hsis, masks=dataset_creator_val.masks,
        augmentation=test_augmentation,
        shuffle_data=False,
    )
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1)
    
    # Test
    dataset_creator_test = DatasetCreator(
        path_hsi=PATH_TEST_UNKNOWN_HSI_DATA,
        path_masks=PATH_TEST_UNKNOWN_MASKS_DATA,
        preprocessing=preprocessing, 
        create_shared_memory=False
    )
    dataset_test = HsiDatasetDataloader(
        hsis=dataset_creator_test.hsis, masks=dataset_creator_test.masks,
        augmentation=test_augmentation,
        shuffle_data=False,
    )
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1)

    net = return_net(arch_type=arch_type, in_f=INPUT_DIM, out_f=NUM_CLASSES)
    net.eval()
    with torch.no_grad():
        _ = net(torch.randn(1, INPUT_DIM, DATA_SHAPE[0], DATA_SHAPE[1]))
    net.train()
    comet_exp = comet_ml.Experiment(
        api_key="your-key",
        workspace="your-workspace",  # Optional
        project_name="your-project-name",  # Optional
    )
    # Random several times
    random_id = [random.randint(0, 1_000_000_000) for _ in range(100)][-1]

    name_exp = f"AttentionV11-WO({KFOLD}) with other (run={number_of_exp})"
    comet_exp.set_name(name_exp)
    comet_exp.add_tag('attention_wo_module_with_other')
    comet_exp.add_tag(KFOLD)
    comet_exp.log_parameter("id_save_folder", random_id)
    comet_exp.log_parameter("arch_type", arch_type)
    comet_exp.log_parameter("max_epochs", max_epochs)
    comet_exp.log_parameter("use pca as input", 'true')
    comet_exp.log_parameter("use cosine lr", 'true')
    comet_exp.log_parameter("additional data prepropcess", 'RustamPreprocess(k=1)')
    comet_exp.log_parameter("batch_size", batch_size)
    comet_exp.log_parameter("start lr", lr)
    comet_exp.log_parameter("cosine(t_0)", T_0)
    comet_exp.log_parameter("cosine(t_mul)", T_mult)
    comet_exp.log_parameter("id_save_folder", random_id)

    model = NnModel(net, nn.CrossEntropyLoss(), experiment=comet_exp,
        T_0=T_0, T_mult=T_mult, lr=lr,
        epoch_list=epoch_list, lr_list=lr_list,
    )

    # saves a checkpoint-file
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"pytorch_li_logs/{random_id}",
        monitor="mean_iou",
        filename="model-{epoch:02d}-{mean_iou:.2f}",
        save_top_k=-1,
        mode="max",
    )


    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=max_epochs,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)
    comet_exp.end()
    # Make sure shared memory is close and unlinked
    # In order to clean memory
    if USE_SHM:
        print('Clear shared memory...')
        dataset_creator_train.close_shm()

    print(f'=======END EXP NUMBER {number_of_exp} ================')


def main():
    for arch_type, params in ARCH_TYPES:
        for exp_params in EXP_DATA_LIST:
            params.update(exp_params)
            for number_of_exp in range(1, N_REPEAT_EXP+1):
                start_exp(
                    number_of_exp=number_of_exp, T_0=T_0_param,
                    T_mult=T_mult_param, arch_type=arch_type, **params
                )


if __name__ == '__main__':
    main()

