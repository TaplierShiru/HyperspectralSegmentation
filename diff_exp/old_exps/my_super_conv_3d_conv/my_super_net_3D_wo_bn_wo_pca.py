#!/usr/bin/env python
# coding: utf-8
# %%

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

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


def collect_prediction_and_target(eval_loader, model):
    target_list = []
    pred_list = []
    
    for in_data_x, val_data in iter(eval_loader):
        preds = model(in_data_x)
        
        target_list.append(val_data)
        pred_list.append(preds)
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
        

def calculate_iou(pred_list, target_list, num_classes=17, loss=None):
    res_list = []
    loss_list = []
    pred_as_mask_list = []
    
    for preds, target in zip(pred_list, target_list):
        if loss is not None:
            loss_list.append(
                loss(torch.unsqueeze(preds, dim=0), torch.unsqueeze(target, dim=0)).cpu().detach().numpy()
            )
        else:
            loss_list.append(None)
        
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
    return res_np, loss_list, pred_as_mask_list


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
        preds = self.model(img)
        loss = self.loss(preds, mask)
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
        metric, loss_list, pred_as_mask_list = calculate_iou(pred_tensor, target_tensor, loss=self.loss)
        
        for batch_idx, (loss_s, metric_s, target_s, pred_s) in enumerate(zip(loss_list, metric, target_tensor, pred_as_mask_list)):
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
                
            d = {f'loss_image_{batch_idx}': torch.tensor(loss_s, dtype=torch.float) }
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
        mean_loss_dict = {
            "mean_loss": torch.tensor(np.asarray(loss_list).mean(), dtype=torch.float),
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

            self.experiment.log_metrics(
                mean_loss_dict,
                epoch=self.current_epoch
            )
        else:
            print(mean_dice_loss_per_class_dict)
            print(mean_dice_loss_dict)
            print(mean_iou_class_dict)
            print(mean_iou_dict)
            print(mean_loss_dict)
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


def preprocessing(imgs, masks):
    with open(f'{PREFIX_INFO_PATH}/data_standartization_params_kfold0.json', 'r') as f:
        data_standartization_params = json.load(f)
    mean = data_standartization_params.get('means')
    std = data_standartization_params.get('stds')
    assert mean is not None and std is not None
    def standartization(img):
        return np.array((img - mean) / std, dtype=np.float32)
    _images = [np.transpose(image, (1, 2, 0)) for image in imgs] #[pca_transformation(image) for image in imgs]
    #_images = [standartization(image) for image in _images]
    _masks = [
        np.expand_dims(
            cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            ,0
        ).astype(np.int64)
        for mask in masks
    ]
    return _images, _masks


# %%


def test_augmentation(image, mask):
    image = TF.to_tensor(image)
    #image = (image - image.min()) / (image.max() - image.min())
    
    mask = torch.from_numpy(mask)
    
    mask = torch.squeeze(mask, 0)
    return image, mask


# %%


def augmentation(image, mask):
    image = TF.to_tensor(image)
    mask = torch.from_numpy(mask)
    angle = T.RandomRotation.get_params((-30, 30))
    image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
    mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
    
    if np.random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    if np.random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    
    #image = (image - image.min()) / (image.max() - image.min())
    mask = torch.squeeze(mask, 0)
    return image, mask



class MySuperNet3D(nn.Module):
    
    def __init__(self, in_f=17, out_f=17, *args):
        super().__init__()
        #self.bn_start = nn.BatchNorm3d(in_f)
        
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(5, 5, 11), stride=(1, 1, 3), padding=(2, 2, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv3d(16, 16, kernel_size=(5, 5, 5), stride=(1, 1, 3), padding=(2, 2, 0))
        self.bn2 = nn.BatchNorm3d(16)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv3d(16, 16, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 0))
        self.bn3 = nn.BatchNorm3d(16)
        self.act3 = nn.ReLU()
        
        self.conv4 = nn.Conv3d(16, 1, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 0))
        self.bn4 = nn.BatchNorm3d(1)
        self.act4 = nn.ReLU()
        
        self.final_conv = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)
    
    def __call__(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.unsqueeze(dim=1)
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
        
        x = self.final_conv(x) 
        x = x.squeeze(dim=1)
        x = x.permute(0, 3, 1, 2)
        return x

# %%

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossCustom(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, alpha=None, gamma=5.5, balance_index=2, smooth=1e-5, size_average=False):
        super(FocalLossCustom, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        self.cel = nn.CrossEntropyLoss(reduction='none')
        self.softmax = nn.Softmax(dim=-1)

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            # N,C,m -> N,m,C
            logit = logit.permute(0, 2, 1).contiguous()
            # N,m,C -> N,m*C
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        
        ce_loss = self.cel(logit, target.view(-1))
        train_conf = self.softmax(logit)
        
        idx = target.cpu().long()
        one_hot_labels  = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_labels  = one_hot_labels.scatter_(1, idx, 1)
        if one_hot_labels.device != logit.device:
            one_hot_labels = one_hot_labels.to(logit.device)
        
        filtered_conf = train_conf * one_hot_labels
        sparce_conf, _ = torch.max(filtered_conf, dim=-1)
        loss = torch.pow((torch.ones_like(sparce_conf) - sparce_conf), self.gamma) * ce_loss
        if self.size_average:
            loss = loss.mean()
        if not self.size_average:
            # Norm by positive
            num_positive = torch.sum(target != self.balance_index)
            loss = loss.sum() / (num_positive + 1e-10)
        else:
            loss = loss.sum()
        return loss


N_REPEAT_EXP = 3

T_0_param = 2
T_mult_param = 1
GAMMAS_LIST = [2.0, 5.5]

NAME2CLASS = {
    MySuperNet3D.__name__:                                         MySuperNet3D,
}

ARCH_TYPES = [
    (MySuperNet3D.__name__,                                       {"batch_size": 4}),
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



def start_exp(number_of_exp: int, T_0: int, T_mult: int, arch_type: str, batch_size: int, gamma: float):

    print(f'=======START EXP NUMBER {number_of_exp} ==============')
    print(f'Params: number_of_exp={number_of_exp}, T_0={T_0} \n' +\
          f'T_mult={T_mult}, arch_type={arch_type}, batch_size={batch_size} \n' +\
          f'gamma={gamma}'
    )
    test_indices = np.load(f'{PREFIX_INFO_PATH}/kfold0_indx_test.npy')
    train_indices = np.load(f'{PREFIX_INFO_PATH}/kfold0_indx_train.npy')

    dataset_train = HsiDataloader(
        PATH_DATA, preprocessing=preprocessing, 
        augmentation=augmentation, indices=train_indices,
        shuffle_data=True
    )
    dataset_test = HsiDataloader(
        PATH_DATA, preprocessing=preprocessing, 
        augmentation=test_augmentation, indices=test_indices
    )

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1)
    
    net = return_net(arch_type=arch_type, in_f=17, out_f=17)
    _ = net(torch.randn(1, 237, 512, 512)).shape

    comet_exp = comet_ml.Experiment(
        api_key="your-key",
        workspace="your-workspace",  # Optional
        project_name="your-project-name",  # Optional
    )
    name_exp = f"MySuperNet3D WithOut BN WoPca (run {number_of_exp}) | LrCosine W weight decay | lower arch | 70ep" +\
               f" | RustamPreprocess(k=1) | makiloss | gamma={gamma} | balance=2" +\
               f"cosine(t_0={T_0},t_mul={T_mult}) | arch_type={arch_type}"
    comet_exp.set_name(name_exp)
    comet_exp.add_tag("conv3d-exp")

    model = NnModel(net, FocalLossCustom(gamma=gamma), experiment=comet_exp,
        T_0=T_0, T_mult=T_mult
    )

    # saves a checkpoint-file
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"pytorch_li_logs/(run={number_of_exp})MySuperNet3D WithOut BN WoPca | _LrCosine W weight decay lower_" +\
                f"arch_50ep_W._RustamPreprocess(k=1)_makiloss_gamma={gamma}_balance=2__"+\
                f"cosine(t_0={T_0},t_mul={T_mult}) | arch_type={arch_type}",
        monitor="mean_iou",
        filename="model-{epoch:02d}-{mean_iou:.2f}",
        save_top_k=-1,
        mode="min",
    )


    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=70,
        check_val_every_n_epoch=2,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    comet_exp.end()

    print(f'=======END EXP NUMBER {number_of_exp} ================')


def main():
    for gamma_val in GAMMAS_LIST:
        for arch_type, params in ARCH_TYPES:
            for number_of_exp in range(1, N_REPEAT_EXP+1):
                start_exp(
                    number_of_exp=number_of_exp, T_0=T_0_param, gamma=gamma_val,
                    T_mult=T_mult_param, arch_type=arch_type, **params
                )


if __name__ == '__main__':
    main()

