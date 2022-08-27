import torch
from torch import nn
import numba as nb
import numpy as np
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# torch.backends.cudnn.benchmark = True
device = 'cuda:0'


@nb.njit#(parallel=True)
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


@nb.njit#(parallel=True)
def merge_parts_into_single_mask(
        preds, pred_mask, h_parts: int, 
        w_parts: int, h_win: int, w_win: int):
    counter = 0

    for h_i in range(h_parts):
        for w_i in range(w_parts):
            pred_mask[:, :,  
                h_i * h_win: (h_i+1) * h_win, 
                w_i * w_win: (w_i+1) * w_win
            ] = preds[counter]
            counter += 1
    return pred_mask



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


arr_data = np.random.randn(1, 17, 512, 512).astype(np.float32)

net = MySuperNetLittleInput(17, 17)
net.to(device=device)
net.eval()

arr_torch = torch.from_numpy(arr_data).to(device=device)

n_1 = 4
n_2 = 200

def test_model(net, arr):
    start = time.time()
    pred_mask = np.zeros(
        (1, 17, 512, 512),
        dtype=np.float32
    )
    for i in range(n_1):
        for j in range(n_2):
            with torch.no_grad():
                arr_cpu = arr.cpu().numpy()
                input_tensors_list = cut_into_parts_model_input(
                    arr_cpu, h_parts=512//8, w_parts=512//8, h_win=8, w_win=8 
                )
                final_input_tensor = np.concatenate(input_tensors_list, axis=0)
                final_input_tensor = torch.from_numpy(final_input_tensor).to(device=device)
                pred_batch = net(final_input_tensor).cpu().numpy()
                final_mask = merge_parts_into_single_mask(
                    preds=pred_batch, pred_mask=pred_mask, h_parts=512//8, 
                    w_parts=512//8, h_win=8, w_win=8
                )
    end = time.time()
    print(f"Time per batch: {(end - start) / (n_1 * n_2)}s  total time={end - start}")

if __name__ == "__main__":
    test_model(net, arr_torch)

