import argparse
import os
import numpy as np

import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import glob
import time

from PIL import Image

from skimage import io

from ulti import Get_DCT_transformation, Tensor2Array, InverseAugmentation

from Burst_Model import Pyramidnet, DCTnet, FilteringNetwork

# load Network
parser = argparse.ArgumentParser(description = 'Track-2 TESTING PHASE')
parser.add_argument('--is_cuda', default = 'cuda:0', type = str)
parser.add_argument('--is_ensemble', default = 1, type = int)

if __name__ == '__main__':
    opt = parser.parse_args()

    # Save folder path
    save_path = 'Track2_Results/'

    # Testing folder path
    path = 'DATA/burst_test/*.png'

    pconst = 'DATA/burst_test/'
    index = len(pconst)

    T_128 = Get_DCT_transformation(128)

    with torch.no_grad():
        # Network weights path
        P_net_path = 'MODEL/BURST/Pixelnet/net_571.pth'
        DCT_net_path = 'MODEL/BURST/DCTnet/net_497.pth'
        Fnet_path = 'MODEL/BURST/Fnet/net_165.pth'

        # Options
        device = opt.is_cuda
        is_ensemble = opt.is_ensemble

        # Load networks
        P_Net = Pyramidnet()

        state_dict = torch.load(P_net_path, map_location=lambda s, l: s)
        P_Net.load_state_dict(state_dict)
        P_Net.to(device)

        DCT_Net = DCTnet()

        state_dict = torch.load(DCT_net_path, map_location=lambda s, l: s)
        DCT_Net.load_state_dict(state_dict)
        DCT_Net.to(device)

        Fnet = FilteringNetwork()

        state_dict = torch.load(Fnet_path, map_location=lambda s, l: s)
        Fnet.load_state_dict(state_dict)
        Fnet.eval()
        Fnet.to(device)

        # Calculate network parameters
        p_params = sum(p.numel() for p in P_Net.parameters() if p.requires_grad)
        dct_params = sum(p.numel() for p in DCT_Net.parameters() if p.requires_grad)
        F_params = sum(p.numel() for p in Fnet.parameters() if p.requires_grad)

        pytorch_total_params = p_params + dct_params + F_params

        print("===> Total trainable params: %d" % (pytorch_total_params))

        # Load testing images
        INPUT = sorted(glob.glob(path))
        size_of_data = len(INPUT) // 7

        # Average computing variable for testing set
        avg_compt_time = 0

        for i in range(size_of_data):

            print('Proceed Image: %d' % (i + 1))

            # Computing variable for per image
            compt_time = 0

            input1 = Image.open(INPUT[i * 7])

            input2 = Image.open(INPUT[i * 7 + 1])

            input3 = Image.open(INPUT[i * 7 + 2])

            input4 = Image.open(INPUT[i * 7 + 3])

            input5 = Image.open(INPUT[i * 7 + 4])

            input6 = Image.open(INPUT[i * 7 + 5])

            input7 = Image.open(INPUT[i * 7 + 6])

            max_k = 1

            if is_ensemble:
                max_k = 8

            for k in range(max_k):

                if k == 0:
                    in1 = input1
                    in2 = input2
                    in3 = input3
                    in4 = input4
                    in5 = input5
                    in6 = input6
                    in7 = input7

                if k == 1:
                    in1 = TF.hflip(input1)
                    in2 = TF.hflip(input2)
                    in3 = TF.hflip(input3)
                    in4 = TF.hflip(input4)
                    in5 = TF.hflip(input5)
                    in6 = TF.hflip(input6)
                    in7 = TF.hflip(input7)

                if k == 2:
                    in1 = TF.vflip(input1)
                    in2 = TF.vflip(input2)
                    in3 = TF.vflip(input3)
                    in4 = TF.vflip(input4)
                    in5 = TF.vflip(input5)
                    in6 = TF.vflip(input6)
                    in7 = TF.vflip(input7)

                if k > 2 and k < 6:
                    in1 = TF.rotate(input1, -90)
                    in2 = TF.rotate(input2, -90)
                    in3 = TF.rotate(input3, -90)
                    in4 = TF.rotate(input4, -90)
                    in5 = TF.rotate(input5, -90)
                    in6 = TF.rotate(input6, -90)
                    in7 = TF.rotate(input7, -90)

                    if k == 4:
                        in1 = TF.hflip(in1)
                        in2 = TF.hflip(in2)
                        in3 = TF.hflip(in3)
                        in4 = TF.hflip(in4)
                        in5 = TF.hflip(in5)
                        in6 = TF.hflip(in6)
                        in7 = TF.hflip(in7)

                    if k == 5:
                        in1 = TF.vflip(in1)
                        in2 = TF.vflip(in2)
                        in3 = TF.vflip(in3)
                        in4 = TF.vflip(in4)
                        in5 = TF.vflip(in5)
                        in6 = TF.vflip(in6)
                        in7 = TF.vflip(in7)

                else:
                    if k == 6:
                        in1 = TF.rotate(input1, 90)
                        in2 = TF.rotate(input2, 90)
                        in3 = TF.rotate(input3, 90)
                        in4 = TF.rotate(input4, 90)
                        in5 = TF.rotate(input5, 90)
                        in6 = TF.rotate(input6, 90)
                        in7 = TF.rotate(input7, 90)

                    if k == 7:
                        in1 = TF.rotate(input1, 180)
                        in2 = TF.rotate(input2, 180)
                        in3 = TF.rotate(input3, 180)
                        in4 = TF.rotate(input4, 180)
                        in5 = TF.rotate(input5, 180)
                        in6 = TF.rotate(input6, 180)
                        in7 = TF.rotate(input7, 180)

                in1 = TF.to_tensor(in1).unsqueeze(0).to(device)
                in2 = TF.to_tensor(in2).unsqueeze(0).to(device)
                in3 = TF.to_tensor(in3).unsqueeze(0).to(device)
                in4 = TF.to_tensor(in4).unsqueeze(0).to(device)
                in5 = TF.to_tensor(in5).unsqueeze(0).to(device)
                in6 = TF.to_tensor(in6).unsqueeze(0).to(device)
                in7 = TF.to_tensor(in7).unsqueeze(0).to(device)

                # forward
                start_time = time.time()

                p = P_Net(in1, in2, in3, in4, in5, in6, in7)
                dct = DCT_Net(in1, in2, in3, in4, in5, in6, in7, T_128)

                fused = Fnet(torch.cat((dct, p), dim=1))

                compt_time = compt_time + (time.time() - start_time)

                if k == 0:
                    fused_res = fused
                    dct_res = dct
                    p_res = p
                else:
                    fused = InverseAugmentation(fused.cpu(), k - 1, device)
                    dct = InverseAugmentation(dct.cpu(), k - 1, device)
                    p = InverseAugmentation(p.cpu(), k - 1, device)

                    fused_res = fused_res + fused
                    dct_res = dct_res + dct
                    p_res = p_res + p

            compt_time = compt_time / max_k
            avg_compt_time = avg_compt_time + compt_time

            fused_res = fused_res / max_k
            dct_res = dct_res / max_k
            p_res = p_res / max_k

            # Final results
            output = fused_res
            out_cpu = output.cpu()

            clean_img = Tensor2Array(out_cpu)

            # Save to folder
            p = INPUT[i * 7]
            image_name = p[index: len(p) - 5]

            Image.fromarray(np.uint8(clean_img * 255.0)).save(save_path + image_name + 'gt.png')

        avg_compt_time = avg_compt_time / size_of_data

        print("Avg. computing time per image: {:.4f} s\n".format(avg_compt_time))
