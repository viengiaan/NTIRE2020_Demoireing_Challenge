import argparse
import os
import numpy as np

import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms.functional as TF

import glob
import time

from PIL import Image

from ulti import Get_DCT_transformation, Tensor2Array, InverseAugmentation

from Single_Model import FilteringNetworkv3, Pyramidnet, DCTnet_3RADB

# load Network
parser = argparse.ArgumentParser(description = 'Track-1 TESTING PHASE')
parser.add_argument('--is_cuda', default = 'cuda:0', type = str)
parser.add_argument('--is_ensemble', default = 1, type = int)

if __name__ == '__main__':
    opt = parser.parse_args()

    # Save folder path
    save_path = 'Track1_Results/'

    # Testing folder path
    path = 'DATA/single_test/*.png'

    pconst = 'DATA/single_test/'
    index = len(pconst)

    T_128 = Get_DCT_transformation(128)

    with torch.no_grad():
        # Network weights path
        P_net_path = 'MODEL/SINGLE/Pixelnet/net_19.pth'
        DCT_net_path = 'MODEL/SINGLE/DCTnet/net_23.pth'
        Fnet_path = 'MODEL/SINGLE/Fnet/net_3.pth'

        # Options
        device = opt.is_cuda
        is_ensemble = opt.is_ensemble

        # Load networks
        P_Net = Pyramidnet()

        state_dict = torch.load(P_net_path, map_location = lambda s, l: s)
        P_Net.load_state_dict(state_dict)
        P_Net.to(device)

        DCT_Net = DCTnet_3RADB()

        state_dict = torch.load(DCT_net_path, map_location = lambda s, l: s)
        DCT_Net.load_state_dict(state_dict)
        DCT_Net.to(device)

        Fnet = FilteringNetworkv3()

        state_dict = torch.load(Fnet_path, map_location = lambda s, l: s)
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

        avg_compt_time = 0

        for i in range(len(INPUT)):

            print('Proceed Image: %d' % (i + 1))

            compt_time = 0

            img = Image.open(INPUT[i])
            input = TF.to_tensor(img).unsqueeze(0).to(device)

            max_k = 1

            if is_ensemble:
                img1 = TF.hflip(img)
                img2 = TF.vflip(img)
                img3 = TF.rotate(img, -90)
                img4 = TF.hflip(img3)
                img5 = TF.vflip(img3)
                img6 = TF.rotate(img, 90)
                img7 = TF.rotate(img, 180)


                input1 = TF.to_tensor(img1).unsqueeze(0).to(device)
                input2 = TF.to_tensor(img2).unsqueeze(0).to(device)
                input3 = TF.to_tensor(img3).unsqueeze(0).to(device)
                input4 = TF.to_tensor(img4).unsqueeze(0).to(device)
                input5 = TF.to_tensor(img5).unsqueeze(0).to(device)
                input6 = TF.to_tensor(img6).unsqueeze(0).to(device)
                input7 = TF.to_tensor(img7).unsqueeze(0).to(device)

                max_k = 8

            for k in range(max_k):
                if k == 0:
                    in_ = input

                if k == 1:
                    in_ = input1

                if k == 2:
                    in_ = input2

                if k == 3:
                    in_ = input3

                if k == 4:
                    in_ = input4

                if k == 5:
                    in_ = input5

                if k == 6:
                    in_ = input6

                if k == 7:
                    in_ = input7

                start_time = time.time()

                p = P_Net(in_)
                dct = DCT_Net(in_, T_128)

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
            p = INPUT[i]
            image_name = p[index : len(p) - 5]

            Image.fromarray(np.uint8(clean_img * 255.0)).save(save_path + image_name + 'gt.png')

        avg_compt_time = avg_compt_time / len(INPUT)

        print("Avg. computing time per image: {:.4f} s\n".format(avg_compt_time))
