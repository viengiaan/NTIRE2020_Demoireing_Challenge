from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms.functional as TF

import numpy as np

from PIL import Image

from math import sqrt
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

def Get_DCT_transformation(N):
    T = torch.zeros((N, N))
    T[0, :] = 1 / np.sqrt(N)

    for i  in range(1, N):
        for j in range(N):
            T[i, j] = np.sqrt(2 / N) * np.cos((2 * j + 1) * i * np.pi / (2 * N))

    return T

def image2tensor(img):
    out = img.swapaxes(0, 2).swapaxes(1, 2)
    out = torch.from_numpy(out * 1.0)
    out = out / 255.0

    channel, height, width = out.size()
    out = torch.reshape(out, (1, channel, height, width))

    return out

def Transform2DCTmatrix(T, inputs):
    (batch, channels, h, w) = inputs.size()
    device = inputs.device

    T_tensor = Variable(T.expand(batch * channels, h, w)).cuda(device)

    T_transpose = torch.t(T)
    T_transpose_tensor = Variable(T_transpose.expand(batch * channels, h, w)).cuda(device)

    x = (inputs * 255.0 - 128).clamp(min=-128, max=127)
    inputs_ = x.view(batch * channels, h, w)
    D = torch.bmm(T_tensor, inputs_)
    D = torch.bmm(D, T_transpose_tensor)
    D = torch.reshape(D, (batch, channels, h, w))

    return D


def DCT2Pixel(T, D):
    (batch, channels, h, w) = D.size()
    device = D.device

    T_tensor = Variable(T.expand(batch * channels, h, w)).cuda(device)

    T_transpose = torch.t(T)
    T_transpose_tensor = Variable(T_transpose.expand(batch * channels, h, w)).cuda(device)

    D = D.view(batch * channels, h, w)
    outputs = torch.bmm(T_transpose_tensor, D)
    outputs = torch.bmm(outputs, T_tensor)
    outputs = torch.reshape(outputs, (batch, channels, h, w))
    outputs = (outputs + 128) / 255.0
    outputs = outputs.clamp(min=0, max=1)

    return outputs

def Tensor2Array(input):
    channel = input.shape[1]
    height = input.shape[2]
    width = input.shape[3]

    out = torch.reshape(input, (channel, height, width))

    clean_img = out.detach().numpy()
    clean_img = clean_img.swapaxes(0, 2).swapaxes(0, 1)
    clean_img[clean_img < 0] = 0
    clean_img[clean_img > 1] = 1

    return clean_img

def InverseAugmentation(inputs, flag, device):
    outputs = Tensor2Array(inputs)
    outputs = Image.fromarray(np.uint8(outputs * 255.0))

    if flag == 0: # horizontal flip
        outputs = TF.hflip(outputs)

    if flag == 1: # vertical flip
        outputs = TF.vflip(outputs)

    if flag == 2: # Rotation -90
        outputs = TF.rotate(outputs, 90)

    if flag == 3: # Rotation -90 + horizontal flip
        outputs = TF.hflip(outputs)
        outputs = TF.rotate(outputs, 90)

    if flag == 4: # Rotation -90 + vertical flip
        outputs = TF.vflip(outputs)
        outputs = TF.rotate(outputs, 90)

    if flag == 5: # Rotation 90
        outputs = TF.rotate(outputs, -90)

    if flag == 6: # Rotation 180
        outputs = TF.rotate(outputs, -180)

    outputs = TF.to_tensor(outputs).unsqueeze(0).to(device)

    return outputs

