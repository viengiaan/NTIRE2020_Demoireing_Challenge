from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import numpy as np

from math import sqrt
from torch.nn.parameter import Parameter
from bam_wo_BN_v2 import CBAM

import torch.nn.functional as F
from torch.autograd import Variable

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, is_SELU = True, kernel_size = 3, dilation = 1, is_BN = False):
        super(make_dense, self).__init__()
        if is_SELU:
            if is_BN:
                self.conv = nn.Sequential(
                    nn.Conv2d(nChannels, growthRate, kernel_size = kernel_size, padding = (kernel_size - 1) // 2 + dilation - 1, stride = 1, dilation = dilation),
                    nn.BatchNorm2d(growthRate),
                    nn.SELU()
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + dilation - 1, stride=1, dilation=dilation),
                    nn.SELU()
                )
        else:
            if is_BN:
                self.conv = nn.Sequential(
                    nn.Conv2d(nChannels, growthRate, kernel_size = kernel_size, padding = (kernel_size - 1) // 2 + dilation - 1, stride = 1, dilation = dilation),
                    nn.BatchNorm2d(growthRate),
                    nn.ReLU()
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + dilation - 1, stride=1, dilation=dilation),
                    nn.ReLU()
                )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = torch.cat((inputs, outputs), dim = 1)

        return outputs

class DenseBlock(nn.Module):
    def __init__(self, in_size, nDenselayer, growthRate, is_SELU = True, is_BN = False):
        super(DenseBlock, self).__init__()
        nChannels_ = in_size
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, is_SELU, is_BN))
            nChannels_ += growthRate

        self.dense_layers = nn.Sequential(*modules)

        if is_SELU:
            if is_BN:
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(nChannels_, in_size, kernel_size = 1),
                    nn.BatchNorm2d(in_size),
                    nn.SELU()
                )
            else:
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(nChannels_, in_size, kernel_size=1),
                    nn.SELU()
                )
        else:
            if is_BN:
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(nChannels_, in_size, kernel_size = 1),
                    nn.BatchNorm2d(in_size),
                    nn.ReLU()
                )
            else:
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(nChannels_, in_size, kernel_size=1),
                    nn.ReLU()
                )

    def forward(self, inputs):
        outputs = self.dense_layers(inputs)
        outputs = self.bottleneck(outputs)

        return outputs

class SA_make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size = 3, dilation = 1, is_SELU = True, is_BN = False):
        super(SA_make_dense, self).__init__()

        if is_SELU:
            if is_BN:
                self.conv = nn.Sequential(
                    nn.Conv2d(nChannels, growthRate, kernel_size = kernel_size, padding = (kernel_size - 1) // 2 + dilation - 1, stride = 1, dilation = dilation),
                    nn.BatchNorm2d(growthRate),
                    nn.SELU()
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + dilation - 1, stride=1, dilation=dilation),
                    nn.SELU()
                )
        else:
            if is_BN:
                self.conv = nn.Sequential(
                    nn.Conv2d(nChannels, growthRate, kernel_size = kernel_size, padding = (kernel_size - 1) // 2 + dilation - 1, stride = 1, dilation = dilation),
                    nn.BatchNorm2d(growthRate),
                    nn.ReLU()
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(nChannels, growthRate, kernel_size = kernel_size, padding = (kernel_size - 1) // 2 + dilation - 1, stride = 1, dilation = dilation),
                    nn.ReLU()
                )

        self.CBAM = CBAM(growthRate, 2, is_SELU = is_SELU, is_BN = is_BN)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.CBAM(outputs)

        outputs = torch.cat((inputs, outputs), dim = 1)

        return outputs

class SADenseBlock(nn.Module):
    def __init__(self, in_size, nDenselayer, growthRate, is_SELU = True, is_BN = False):
        super(SADenseBlock, self).__init__()
        nChannels_ = in_size
        modules = []
        for i in range(nDenselayer):
            modules.append(SA_make_dense(nChannels_, growthRate, is_SELU = is_SELU, is_BN = is_BN))
            nChannels_ += growthRate

        self.dense_layers = nn.Sequential(*modules)

        if is_SELU:
            if is_BN:
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(nChannels_, in_size, kernel_size = 1),
                    nn.BatchNorm2d(in_size),
                    nn.SELU()
                )
            else:
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(nChannels_, in_size, kernel_size = 1),
                    nn.SELU()
                )
        else:
            if is_BN:
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(nChannels_, in_size, kernel_size = 1),
                    nn.BatchNorm2d(in_size),
                    nn.ReLU()
                )
            else:
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(nChannels_, in_size, kernel_size = 1),
                    nn.ReLU()
                )

    def forward(self, inputs):
        outputs = self.dense_layers(inputs)
        outputs = self.bottleneck(outputs)

        return outputs

class RSADB(nn.Module):
    def __init__(self, in_size, nDenselayer, growthRate, dilation = 1, is_SELU = True, is_BN = False):
        super(RSADB, self).__init__()
        nChannels_ = in_size
        modules  = []
        for i in range(nDenselayer):
            modules.append(SA_make_dense(nChannels_, growthRate, dilation = dilation, is_SELU = is_SELU, is_BN = is_BN))
            nChannels_ += growthRate

        self.dense_layers = nn.Sequential(*modules)

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(nChannels_, in_size, kernel_size = 1, padding = 0, stride = 1)
        )


    def forward(self, inputs):
        outputs = self.dense_layers(inputs)

        outputs = self.conv_1x1(outputs)

        return outputs + inputs

class Transform2DCT(nn.Module):
    def __init__(self):
        super(Transform2DCT, self).__init__()

    def forward(self, inputs, T):
        (batch, channels, h, w) = inputs.size()
        device = inputs.device

        T_tensor = T.expand(batch * channels, h, w).cuda(device)

        T_transpose = torch.t(T)
        T_transpose_tensor = T_transpose.expand(batch * channels, h, w).cuda(device)

        inputs_ = inputs.view(batch * channels, h, w)
        D = torch.bmm(T_tensor, inputs_)
        D = torch.bmm(D, T_transpose_tensor)

        outputs = D.view(batch, channels, h, w)

        return outputs

class InverseDCT(nn.Module):
    def __init__(self):
        super(InverseDCT, self).__init__()

    def forward(self, inputs, T):
        (batch, channels, h, w) = inputs.size()
        device = inputs.device

        T_tensor = T.expand(batch * channels, h, w).cuda(device)

        T_transpose = torch.t(T)
        T_transpose_tensor = T_transpose.expand(batch * channels, h, w).cuda(device)

        inputs_ = inputs.view(batch * channels, h, w)
        outputs = torch.bmm(T_transpose_tensor, inputs_)
        outputs = torch.bmm(outputs, T_tensor)
        outputs = outputs.view(batch, channels, h, w)

        return outputs

class BasicConv(nn.Module):
    def __init__(self, in_size, out_size, kernel = 3, padding = 1, stride = 1, is_SELU = True, is_BN = False):
        super(BasicConv, self).__init__()

        if is_SELU:
            if is_BN:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size = kernel, padding = padding, stride = stride),
                    nn.BatchNorm2d(out_size),
                    nn.SELU()
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size = kernel, padding = padding, stride = stride),
                    nn.SELU()
                )
        else:
            if is_BN:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size = kernel, padding = padding, stride = stride),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU()
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size = kernel, padding = padding, stride = stride),
                    nn.ReLU()
                )

    def forward(self, inputs):
        outputs = self.conv(inputs)

        return outputs

########################################################################## 28/02
class Sobel_Grads(nn.Module):
    def __init__(self):
        super(Sobel_Grads, self).__init__()

        c = 3

        gx = np.array([[1., 2. , 1.], [0., 0., 0.], [-1., -2. , -1.]], dtype='float32')
        self.gx = Variable(torch.from_numpy(gx).expand(c, 1, 3, 3).contiguous(), requires_grad = False)

        gy = np.array([[1., 0. , -1.], [2., 0., -2.], [1., 0. , -1.]], dtype='float32')
        self.gy = Variable(torch.from_numpy(gy).expand(c, 1, 3, 3).contiguous(), requires_grad=False)


    def forward(self, inputs):
        device = inputs.device

        self.gx = self.gx.to(device)
        self.gy = self.gy.to(device)

        c = 3
        Gx = F.conv2d(inputs, self.gx, padding = 1, groups = c)
        Gy = F.conv2d(inputs, self.gy, padding = 1, groups = c)

        magnitude = torch.sqrt(Gx * Gx + Gy * Gy)

        return magnitude

class DynamicFilterNetwork(nn.Module):
    def __init__(self):
        super(DynamicFilterNetwork, self).__init__()
        filters = [32, 64, 128, 256, 512]

        self.encode = nn.Sequential(
            BasicConv(6, 32),
            SADenseBlock(32, 5, 16)
        )

        self.conv = nn.Conv2d(32, 6, kernel_size = 1)

    def forward(self, inputs):

        encode = self.encode(inputs)

        conv = self.conv(encode)

        return conv

class LocalFiltering(nn.Module):
    def __init__(self):
        super(LocalFiltering, self).__init__()
        self.filter_size = 5
        self.stride = 1
        self.pad = 0

    def forward(self, inputs, filters):
        device = inputs.device

        filter_localexpand_np = torch.reshape(torch.eye(30, 30), (6, 6, 5, 5))
        filter_localexpand_np = filter_localexpand_np.to(device)

        input_localexpanded = F.conv2d(inputs, filter_localexpand_np, stride = 1, padding = 2)

        output = input_localexpanded * filters
        output = torch.sum(output, dim = 1)

        return output

################################################################################## Pyramid on Pixel
class PS_Upsample(nn.Module):
    def __init__(self, in_size, is_SELU = True, is_BN = True):
        super(PS_Upsample, self).__init__()

        out_size = in_size * 2

        if is_SELU:
            if is_BN:
                self.up = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1),
                    nn.BatchNorm2d(out_size),
                    nn.SELU(),
                    nn.PixelShuffle(2)
                )
            else:
                self.up = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1),
                    nn.SELU(),
                    nn.PixelShuffle(2)
                )
        else:
            if is_BN:
                self.up = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(),
                    nn.PixelShuffle(2)
                )
            else:
                self.up = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1),
                    nn.ReLU(),
                    nn.PixelShuffle(2)
                )

    def forward(self, inputs):
        outputs = self.up(inputs)

        return outputs

############################################################################## 28/02/2020
class AttentionModule(nn.Module):
    def __init__(self, filters):
        super(AttentionModule, self).__init__()

        self.att11 = nn.Conv2d(filters * 2, filters * 2, kernel_size = 3, padding = 1)
        self.att12 = nn.Conv2d(filters * 2, filters, kernel_size = 3, padding = 1)

        self.sigmoid = nn.Sigmoid()
        self.af = nn.SELU()

    def forward(self, input1, input2):

        p1_ = torch.cat((input1, input2), dim = 1)
        p1_A = self.af(self.att11(p1_))
        p1_A = self.att12(p1_A)
        p1_A = self.sigmoid(p1_A)
        p1 = input1 * p1_A

        return p1

############################################################################## 14/03/2020
class AttetionNetwork(nn.Module):
    def __init__(self):
        super(AttetionNetwork, self).__init__()
        filters = 32

        self.Aconv1 = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, padding=1),
            nn.SELU()
        )

        self.Aconv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)

        self.conv1x1 = nn.Conv2d(filters * 2, filters, kernel_size = 1)

        self.Extract_feats1 = nn.Sequential(
            nn.Conv2d(filters * 7, filters, kernel_size = 1),
            nn.SELU(),
            RSADB(filters, 5, 16)
        )

        self.Extract_feats2 = nn.Sequential(
            nn.Conv2d(filters * 7, filters, kernel_size = 1),
            nn.SELU(),
            RSADB(filters, 5, 16)
        )

        self.att1 = AttentionModule(filters)
        self.att2 = AttentionModule(filters)
        self.att3 = AttentionModule(filters)
        self.att5 = AttentionModule(filters)
        self.att6 = AttentionModule(filters)
        self.att7 = AttentionModule(filters)

    def forward(self, input1, input2, input3, input4, input5, input6, input7):

        int_feats_01 = self.Aconv1(input1)
        int_feats_02 = self.Aconv1(input2)
        int_feats_03 = self.Aconv1(input3)
        int_feats_04 = self.Aconv1(input4)
        int_feats_05 = self.Aconv1(input5)
        int_feats_06 = self.Aconv1(input6)
        int_feats_07 = self.Aconv1(input7)

        int_feats = torch.cat((int_feats_01, int_feats_02, int_feats_03, int_feats_04, int_feats_05, int_feats_06, int_feats_07), dim = 1)

        global_features1 = self.Extract_feats1(int_feats)

        feats_01 = self.conv1x1(torch.cat((int_feats_01, global_features1), dim=1))
        feats_02 = self.conv1x1(torch.cat((int_feats_02, global_features1), dim=1))
        feats_03 = self.conv1x1(torch.cat((int_feats_03, global_features1), dim=1))

        feats_05 = self.conv1x1(torch.cat((int_feats_05, global_features1), dim=1))
        feats_06 = self.conv1x1(torch.cat((int_feats_06, global_features1), dim=1))
        feats_07 = self.conv1x1(torch.cat((int_feats_07, global_features1), dim=1))

        p1 = self.att1(feats_01, int_feats_04)
        p2 = self.att2(feats_02, int_feats_04)
        p3 = self.att3(feats_03, int_feats_04)

        p5 = self.att5(feats_05, int_feats_04)
        p6 = self.att6(feats_06, int_feats_04)
        p7 = self.att7(feats_07, int_feats_04)

        global_features2 = self.Extract_feats2(torch.cat((p1, p2, p3, int_feats_04, p5, p6, p7), dim = 1))

        outputs = self.Aconv2(global_features2)

        return outputs

class DCTnet(nn.Module):
    def __init__(self):
        super(DCTnet, self).__init__()
        filters = [32, 64]
        ####### DCT
        self.Trans2DCT = Transform2DCT()
        self.IDCT = InverseDCT()

        self.dct_AN = AttetionNetwork()

        self.dct_features = BasicConv(filters[0], filters[0])
        self.SADB1 = SADenseBlock(filters[0], 5, 16)
        self.RSADB1 = RSADB(filters[0], 5, 32)
        self.RSADB2 = RSADB(filters[0], 5, 32)
        self.RSADB3 = RSADB(filters[0], 5, 32)
        self.RSADB4 = RSADB(filters[0], 5, 32)
        self.RSADB5 = RSADB(filters[0], 5, 32)

        self.dct_conv = nn.Conv2d(filters[0], 3, kernel_size = 3, padding = 1)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, DCT_128):
        ### DCT
        dct1 = self.Trans2DCT(inputs1, DCT_128)
        dct2 = self.Trans2DCT(inputs2, DCT_128)
        dct3 = self.Trans2DCT(inputs3, DCT_128)
        dct4 = self.Trans2DCT(inputs4, DCT_128)
        dct5 = self.Trans2DCT(inputs5, DCT_128)
        dct6 = self.Trans2DCT(inputs6, DCT_128)
        dct7 = self.Trans2DCT(inputs7, DCT_128)

        dct = self.dct_AN(dct1, dct2, dct3, dct4, dct5, dct6, dct7)

        dct_features = self.dct_features(dct)
        SADB1 = self.SADB1(dct_features)
        RSADB1 = self.RSADB1(SADB1)
        RSADB2 = self.RSADB2(RSADB1)
        RSADB3 = self.RSADB3(RSADB2)
        RSADB4 = self.RSADB4(RSADB3)
        RSADB5 = self.RSADB5(RSADB4)

        dct_res = self.dct_conv(RSADB5)
        dct_res = self.IDCT(dct_res, DCT_128)

        return dct_res

class Pyramidnet(nn.Module):
    def __init__(self):
        super(Pyramidnet, self).__init__()
        filters = [32, 64, 128, 256, 512]

        self.p_AN = AttetionNetwork()

        ##### Pixel domain
        self.p_128 = BasicConv(filters[0], filters[0])
        self.p_64 = BasicConv(filters[0], filters[1], kernel=2, padding=0, stride=2)
        self.p_32 = BasicConv(filters[1], filters[2], kernel=2, padding=0, stride=2)
        self.p_16 = BasicConv(filters[2], filters[3], kernel=2, padding=0, stride=2)
        self.p_8 = BasicConv(filters[3], filters[4], kernel=2, padding=0, stride=2)

        self.p_1x1_128 = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], kernel_size=1),
            nn.SELU()
        )

        self.p_1x1_64 = nn.Sequential(
            nn.Conv2d(filters[1] * 2, filters[1], kernel_size=1),
            nn.SELU()
        )

        self.p_1x1_32 = nn.Sequential(
            nn.Conv2d(filters[2] * 2, filters[2], kernel_size=1),
            nn.SELU()
        )

        self.p_1x1_16 = nn.Sequential(
            nn.Conv2d(filters[3] * 2, filters[3], kernel_size=1),
            nn.SELU()
        )

        self.p_DB_128 = nn.Sequential(
            SADenseBlock(filters[0], 5, 16),
            RSADB(filters[0], 10, 32)
        )

        self.p_DB_64 = nn.Sequential(
            SADenseBlock(filters[1], 5, 16),
            RSADB(filters[1], 10, 32)
        )

        self.p_DB_32 = nn.Sequential(
            SADenseBlock(filters[2], 5, 16),
            RSADB(filters[2], 10, 32)
        )

        self.p_DB_16 = nn.Sequential(
            SADenseBlock(filters[3], 5, 16),
            RSADB(filters[3], 5, 32)
        )

        self.p_DB_8 = nn.Sequential(
            SADenseBlock(filters[4], 5, 16),
            RSADB(filters[4], 5, 32)
        )

        self.p_up_8 = PS_Upsample(filters[4])
        self.p_up_16 = PS_Upsample(filters[3])
        self.p_up_32 = PS_Upsample(filters[2])
        self.p_up_64 = PS_Upsample(filters[1])

        self.p_3x3_128 = nn.Conv2d(filters[0], 3, kernel_size=3, padding=1)

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7):

        p_AN = self.p_AN(inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7)

        # Pixel domain
        p_inputs = p_AN

        p_128 = self.p_128(p_inputs)
        p_64 = self.p_64(p_128)
        p_32 = self.p_32(p_64)
        p_16 = self.p_16(p_32)
        p_8 = self.p_8(p_16)

        ### 8x8
        p_DB_8 = self.p_DB_8(p_8)
        p_up_8 = self.p_up_8(p_DB_8)

        ### 16x16
        p_DB_16 = self.p_1x1_16(torch.cat((p_16, p_up_8), dim = 1))
        p_DB_16 = self.p_DB_16(p_DB_16)
        p_up_16 = self.p_up_16(p_DB_16)

        ### 32x32
        p_DB_32 = self.p_1x1_32(torch.cat((p_32, p_up_16), dim = 1))
        p_DB_32 = self.p_DB_32(p_DB_32)
        p_up_32 = self.p_up_32(p_DB_32)

        ### 64x64
        p_DB_64 = self.p_1x1_64(torch.cat((p_64, p_up_32), dim = 1))
        p_DB_64 = self.p_DB_64(p_DB_64)
        p_up_64 = self.p_up_64(p_DB_64)

        ### 128x128
        p_DB_128 = self.p_1x1_128(torch.cat((p_128, p_up_64), dim = 1))
        p_DB_128 = self.p_DB_128(p_DB_128)

        p_res = self.p_3x3_128(p_DB_128)

        return p_res

class FilteringNetwork(nn.Module):
    def __init__(self):
        super(FilteringNetwork, self).__init__()
        self.LocalFilter = LocalFiltering()
        self.F_R = DynamicFilterNetwork()
        self.F_G = DynamicFilterNetwork()
        self.F_B = DynamicFilterNetwork()

    def forward(self, inputs):
        F_R = self.F_R(inputs)
        F_G = self.F_G(inputs)
        F_B = self.F_B(inputs)

        R = self.LocalFilter(inputs, F_R).unsqueeze(1)
        G = self.LocalFilter(inputs, F_G).unsqueeze(1)
        B = self.LocalFilter(inputs, F_B).unsqueeze(1)

        outputs = torch.cat((R, G, B), dim = 1)

        return outputs
