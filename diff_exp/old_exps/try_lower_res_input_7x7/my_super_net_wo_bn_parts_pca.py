#!/usr/bin/env python
# coding: utf-8
# %%

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import sys
sys.path.append('/home/rustam/hyperspecter_segmentation/makitorch')
sys.path.append('/home/rustam/hyperspecter_segmentation/')

PREFIX_INFO_PATH = '/home/rustam/hyperspecter_segmentation/danil_cave/kfolds_data/kfold0'
PATH_DATA = '/raid/rustam/hyperspectral_dataset/new_cropped_hsi_data'


from makitorch import *

import numpy as np
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


class HsiDataloaderCutter(torch.utils.data.IterableDataset):
    def __init__(
            self, 
            data_path: str,
            preprocessing: Optional[Union[DataPreprocessor, Callable]] = BaseDataPreprocessor(),
            augmentation: Optional[Union[DataAugmentator, Callable]] = BaseDataAugmentor(),
            indices = None,
            shuffle_data=False,
            cut_window=(8, 8)
        ):
        super().__init__()
        self.shuffle_data = shuffle_data
        self.dataset = HsiDataset(data_path)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.cut_window = cut_window
        
        self.images = []
        self.masks = []
        
        for idx, data_point in enumerate(self.dataset.data_iterator(opened=True, shuffle=False)):
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
        
        if self.preprocessing is not None:
            self.images, self.masks = self.preprocessing(self.images, self.masks, cut_window=cut_window)
    
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

    def __iter__(self):
        if self.shuffle_data:
            self.images, self.masks = shuffle(self.images, self.masks)
        
        for image, mask in zip(self.images, self.masks):
            yield self.augmentation(image, mask, self.cut_window)




def clear_metric_calculation(final_metric, target_t, pred_t, num_classes=17):
    """
    
    Parameters
    ----------
    final_metric: torch.Tensor
        Tensor with shape (N, C)
    target_t: torch.Tensor or list
        Tensor with shape (N, 1, H, W)
    pred_t: torch.Tensor or list
        Tensor with shape (N, C, H, W)
    
    """
    # For each image
    final_metric_dict = dict([
        (str(i), []) for i in range(num_classes)
    ])
    for metric_s, target_t_s, pred_t_s in zip(final_metric, target_t, pred_t):
        unique_indx_target = torch.unique(target_t_s) 
        unique_indx_pred = torch.unique(pred_t_s)
        for i in range(num_classes):
            if i in unique_indx_target or i in unique_indx_pred:
                final_metric_dict[str(i)].append(metric_s[i])
    
    mean_per_class_metric = [
        sum(final_metric_dict[str(i)]) / len(final_metric_dict[str(i)])
        if len(final_metric_dict[str(i)]) != 0
        else 0.0
        for i in range(num_classes)
    ] 
    mean_metric = sum(mean_per_class_metric) / len(mean_per_class_metric)
    return mean_per_class_metric, mean_metric


def matrix2onehot(matrix, num_classes=17):
    matrix = matrix.copy().reshape(-1)
    one_hoted = np.zeros((matrix.size, num_classes))
    one_hoted[np.arange(matrix.size),matrix] = 1
    return one_hoted


def collect_prediction_and_target(eval_loader, model, cut_window=(8, 8), image_shape=(512, 512), num_classes=17):
    target_list = []
    pred_list = []
    
    for in_data_x, val_data in iter(eval_loader):
        batch_size = in_data_x.shape[0]
        pred_mask = torch.zeros(
            (batch_size, num_classes, image_shape[0], image_shape[1]),
            dtype=in_data_x.dtype, device=in_data_x.device
        )
        # Take prediction from each window
        for w_i in range(image_shape[1] // cut_window[1]):
            for h_i in range(image_shape[0] // cut_window[0]):
                img_part = in_data_x[:, :, 
                    h_i * cut_window[0]: (h_i+1) * cut_window[0],
                    w_i * cut_window[1]: (w_i+1) * cut_window[1]
                ]
                pred = model(img_part) # (N, num_classes) -> (N, num_classes, 1, 1)
                pred = pred.unsqueeze(dim=-1).unsqueeze(dim=-1)
                pred_mask[:, :,
                    h_i * cut_window[0]: (h_i+1) * cut_window[0],
                    w_i * cut_window[1]: (w_i+1) * cut_window[1]
                ] = pred
        
        target_list.append(val_data)
        pred_list.append(pred_mask)
    return (torch.cat(pred_list, dim=0), 
            torch.cat(target_list, dim=0)
    )


def list_target_to_onehot(target_tensor, num_classes=17):
    one_hoted_list = []
    for target in target_tensor:
        target =  np.squeeze(target.cpu().detach().numpy())
        h,w = target.shape
        target = matrix2onehot(target, num_classes=num_classes)
        target = target.reshape(h, w, -1)
        target = np.transpose(target, [2, 0, 1])
        one_hoted_list.append(target)
    return torch.from_numpy(np.stack(one_hoted_list, axis=0))
        

def calculate_iou(pred_list, target_list, num_classes=17):
    res_list = []
    pred_as_mask_list = []
    
    for preds, target in zip(pred_list, target_list):
        
        preds = nn.functional.softmax(preds, dim=0).cpu().detach().numpy()
        preds = np.squeeze(np.argmax(preds, axis=0))
        pred_as_mask_list.append(preds)
        
        target = np.squeeze(target.cpu().detach().numpy())
        
        preds_one_hoted = matrix2onehot(preds, num_classes=num_classes)
        target_one_hoted = matrix2onehot(target, num_classes=num_classes)
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


# %%


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if getattr(m, 'bias') is not None:
            m.bias.data.fill_(0.01)


# %%


class WeightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=w.clamp(0, 1)
            module.weight.data=w


# %%


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, fs_weight, preds, mask):
        return self.ce(preds, mask) + torch.sum(1 - (torch.abs(fs_weight) / 0.99 - 1) ** 2)


# %%


class NnModel(pl.LightningModule):
    def __init__(
            self, model, loss,
            T_0=10, T_mult=2, experiment=None, enable_image_logging=True):
        super().__init__()
        self.model = model
        self.loss = loss
        self.experiment = experiment
        self.enable_image_logging = enable_image_logging
        #self.weight_contraint_function = WeightConstraint()

        self.T_0 = T_0
        self.T_mult = T_mult

    def _custom_histogram_adder(self):
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)
            
    def forward(self, x):
        out = self.model(x)
        return out
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=1e-3
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.T_0, T_mult=self.T_mult, eta_min=0
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    def training_step(self, train_batch, batch_idx):
        img, mask = train_batch
        preds = self.model(img) # (N, C)
        loss = self.loss(preds, mask) # (N,)
        self.log('train_loss', loss)
        if self.experiment is not None:
            self.experiment.log_metric("train_loss", loss, epoch=self.current_epoch, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        return batch
    
    def validation_epoch_end(self, outputs):
        print('Size epoch end input: ', len(outputs))
        
        pred_tensor, target_tensor = collect_prediction_and_target(outputs, self.model)
        target_one_hotted_tensor = list_target_to_onehot(target_tensor)
        dice_loss_val = dice_loss(pred_tensor, target_one_hotted_tensor, dim=[0, 2, 3], use_softmax=True, softmax_dim=1)
        metric, pred_as_mask_list = calculate_iou(pred_tensor, target_tensor)
        
        for batch_idx, (metric_s, target_s, pred_s) in enumerate(zip(metric, target_tensor, pred_as_mask_list)):
            if self.enable_image_logging:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                sns.heatmap(pred_s, ax=ax1, vmin=0, vmax=17)
                sns.heatmap(target_s.cpu().detach().numpy(), ax=ax2, vmin=0, vmax=17)
                fig.savefig('temp_fig.png')
                plt.close(fig)

    #             trainer.logger.experiment.log_histogram_3d(
    #                 self.model.features_selection.weight.detach().cpu().numpy(),
    #                 name='band-selection layer',
    #                 step=self.global_step
    #             )
                if self.experiment is not None:
                    # For Comet logger
                    self.experiment.log_image(
                        'temp_fig.png', name=f'{batch_idx}', 
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
                np.asarray(pred_as_mask_list).reshape(-1)
            )
            
        mean_per_class_metric, mean_metric = clear_metric_calculation(metric, target_tensor, pred_tensor)
        mean_dice_loss_per_class_dict = {
            f"mean_dice_loss_per_class_{i}": torch.tensor(d_l, dtype=torch.float)
            for i, d_l in enumerate(dice_loss_val)
        }
        mean_dice_loss_dict = {
            f"mean_dice_loss": torch.tensor(dice_loss_val.mean(), dtype=torch.float)
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


# %%


device = 'cuda:0'


# %%


pca_explained_variance = np.load(f'{PREFIX_INFO_PATH}/kfold0_PcaExplainedVariance_.npy')
pca_mean = np.load(f'{PREFIX_INFO_PATH}/kfold0_PcaMean.npy')
pca_components = np.load(f'{PREFIX_INFO_PATH}/kfold0_PcaComponents.npy')


# %%


def pca_transformation(x):
    x_t = np.reshape(x, (x.shape[0], -1)) # (C, H, W) -> (C, H * W)
    x_t = np.swapaxes(x_t, 0, 1) # (C, H * W) -> (H * W, C)
    x_t = x_t - pca_mean
    x_t = np.dot(x_t, pca_components.T) / np.sqrt(pca_explained_variance)
    return np.reshape(x_t, (x.shape[1], x.shape[2], pca_components.shape[0])).astype(np.float32) # (H, W, N)


# %%

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


def preprocessing(imgs, masks, cut_window=None):
    with open(f'{PREFIX_INFO_PATH}/data_standartization_params_kfold0.json', 'r') as f:
        data_standartization_params = json.load(f)
    mean = data_standartization_params.get('means')
    std = data_standartization_params.get('stds')
    assert mean is not None and std is not None
    def standartization(img):
        return np.array((img - mean) / std, dtype=np.float32)
    _images = [pca_transformation(image) for image in imgs]
    _images = [standartization(image) for image in _images]
    _masks = [
        np.expand_dims(
            cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            ,0
        ).astype(np.int64)
        for mask in masks
    ]
    if cut_window is not None:
        _masks = [mask2class(mask) for mask in _masks]
    return _images, _masks


# %%


def test_augmentation(image, mask, *args):
    image = TF.to_tensor(image)
    #image = (image - image.min()) / (image.max() - image.min())
    
    mask = torch.from_numpy(mask)
    
    mask = torch.squeeze(mask, 0)
    return image, mask


# %%


def augmentation(image, mask, cut_window=None, *args):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(mask)
    angle = T.RandomRotation.get_params((-30, 30))
    image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
    if cut_window is None:
        mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
    
    if np.random.random() > 0.5:
        image = TF.hflip(image)
        if cut_window is None:
            mask = TF.hflip(mask)

    if np.random.random() > 0.5:
        image = TF.vflip(image)
        if cut_window is None:
            mask = TF.vflip(mask)
    
    #image = (image - image.min()) / (image.max() - image.min())
    mask = torch.squeeze(mask, 0)
    return image, mask



class MySuperNetLittleInput(nn.Module):
    
    def __init__(self, in_f=17, out_f=17, *args):
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

        self.pooling = nn.AvgPool2d(2, stride=1, padding=0)
        # (1, 64, 7, 7)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(64 * 7 * 7, 17, bias=False)
    
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

        x = self.pooling(x)
        x = x.view(x.shape[0], -1) # (N, ...)
        x = self.dropout(x)
        x = self.linear(x)
        return x


N_REPEAT_EXP = 3

T_0_param = 2
T_mult_param = 1

NAME2CLASS = {
    MySuperNetLittleInput.__name__:             MySuperNetLittleInput,
}

ARCH_TYPES = [
    (MySuperNetLittleInput.__name__,            {"batch_size": 40}),
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



def start_exp(number_of_exp: int, T_0: int, T_mult: int, arch_type: str, batch_size: int):

    print(f'=======START EXP NUMBER {number_of_exp} ==============')
    print(f'Params: number_of_exp={number_of_exp}, T_0={T_0} \n' +\
          f'T_mult={T_mult}, arch_type={arch_type}, batch_size={batch_size} \n'
    )
    test_indices = np.load(f'{PREFIX_INFO_PATH}/kfold0_indx_test.npy')
    train_indices = np.load(f'{PREFIX_INFO_PATH}/kfold0_indx_train.npy')

    dataset_train = HsiDataloaderCutter(
        PATH_DATA, preprocessing=preprocessing, 
        augmentation=augmentation, indices=train_indices,
        shuffle_data=True
    )
    dataset_test = HsiDataloaderCutter(
        PATH_DATA, preprocessing=preprocessing, 
        augmentation=test_augmentation, indices=test_indices,
        cut_window=None
    )

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1)
    
    net = return_net(arch_type=arch_type, in_f=17, out_f=17)
    _ = net(torch.randn(1, 17, 8, 8)).shape

    comet_exp = comet_ml.Experiment(
        api_key="your-key",
        workspace="your-workspace",  # Optional
        project_name="your-project-name",  # Optional
    )
    name_exp = f"MySuperNetLittleInput W PCA (run {number_of_exp}) | LrCosine W weight decay | lower arch | 10ep" +\
               f" | RustamPreprocess(k=1) | CE" +\
               f"cosine(t_0={T_0},t_mul={T_mult}) | arch_type={arch_type}"
    comet_exp.set_name(name_exp)
    comet_exp.add_tag("conv3d-exp")

    model = NnModel(net, nn.CrossEntropyLoss(), experiment=comet_exp,
        T_0=T_0, T_mult=T_mult
    )

    # saves a checkpoint-file
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"pytorch_li_logs/(run={number_of_exp})MySuperNetLittleInput W PCA | _LrCosine W weight decay lower_" +\
                f"arch_10ep._RustamPreprocess(k=1)_CE"+\
                f"cosine(t_0={T_0},t_mul={T_mult}) | arch_type={arch_type}",
        monitor="mean_iou",
        filename="model-{epoch:02d}-{mean_iou:.2f}",
        save_top_k=-1,
        mode="min",
    )


    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=10,
        check_val_every_n_epoch=2,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    comet_exp.end()

    print(f'=======END EXP NUMBER {number_of_exp} ================')


def main():
    for arch_type, params in ARCH_TYPES:
        for number_of_exp in range(1, N_REPEAT_EXP+1):
            start_exp(
                number_of_exp=number_of_exp, T_0=T_0_param,
                T_mult=T_mult_param, arch_type=arch_type, **params
            )


if __name__ == '__main__':
    main()

