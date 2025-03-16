import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import cv2
import gc
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import numpy as np
from models import CrossEncoder
from models.regressor import regressor
from PIL import Image
import matplotlib.pyplot as plt
import os
from dataloaders.data_loader_FreeGaze2 import ImagerLoader
from sklearn import metrics

growth_rate = 32
z_dim_app = 64
z_dim_gaze = 32
decoder_input_c = 32

network = CrossEncoder(
    growth_rate=growth_rate,
    z_dim_app=z_dim_app,
    z_dim_gaze=z_dim_gaze,
    decoder_input_c=decoder_input_c,
)
pretrain_dict = torch.load('/mnt/traffic/home/jxt/3ndPaper-Allworks/1-ProposedWork/Cross-Encoder/backup/exp1/checkpoints/at_step_0372599.pth.tar')
network.load_state_dict(pretrain_dict)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def send_data_dict_to_gpu(data):
    for k in data:
        v = data[k]
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data


def recover_images(x):
    x = x.cpu().detach().numpy()
    x = x * 255.0
    x = np.clip(x, 0, 255)
    x = np.transpose(x, [0, 2, 3, 1])  # CHW to HWC
    x = x.astype(np.uint8)
    x = x[:, :, :, ::-1]  # RGB to BGR for OpenCV
    return x



def eye_loader(path):
    try:
        im = Image.open(path)
        return im
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")



network = network.to(device)
network.eval()

data_root = '/mnt/traffic/home/jxt/3ndPaper-Allworks/3-Datasets/test_dataset'
batch_size = 1

testset = ImagerLoader(data_root, transforms.Compose([
    transforms.Resize((128, 64)), transforms.ToTensor(),  # image_normalize,
]))
test = torch.utils.data.DataLoader(testset,
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=32, pin_memory=True)

print(len(testset))
data_iterator = iter(test)


image_path = 'images/exp1/'

print('Let\'s start!!!')
for input_dict in test:
    name1l = str(input_dict['name_1_l'])
    name1l = name1l.rstrip(".jpg']").split('/')[-1]
    name2l = str(input_dict['name_2_l'])
    name2l = name1l.rstrip(".jpg']").split('/')[-1]


    input_dict = send_data_dict_to_gpu(input_dict)
    output_dict = network(input_dict)

    output_images = recover_images(torch.cat((output_dict['image_hat_1_l'], output_dict['image_hat_1_r']), dim=3))
    input_images = recover_images(torch.cat((input_dict['img_1_l'], input_dict['img_1_r']), dim=3))
    cv2.imwrite(os.path.join(image_path, 'decimg_' + name1l + '_1.jpg'), output_images[0])
    cv2.imwrite(os.path.join(image_path, 'orgimg_' + name1l + '_1.jpg'), input_images[0])
    output_images = recover_images(torch.cat((output_dict['image_hat_2_l'], output_dict['image_hat_2_r']), dim=3))
    input_images = recover_images(torch.cat((input_dict['img_2_l'], input_dict['img_2_r']), dim=3))
    cv2.imwrite(os.path.join(image_path, 'decimg_' + name2l + '_2.jpg'), output_images[0])
    cv2.imwrite(os.path.join(image_path, 'orgimg_' + name2l + '_2.jpg'), input_images[0])




