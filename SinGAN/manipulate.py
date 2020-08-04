from __future__ import print_function
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments

def SinGAN_generate(Gs,Zs,reals1, reals2,NoiseAmp,opt,in_s=None, scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=50):
    #if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals1[0].shape, 0, device=opt.device)
    images_cur = []

    assert len(reals1) == len(Gs)

    for G,(Z_opt1, Z_opt2),noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        # assumption: same size
        nzx = (Z_opt1.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt1.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            image1_weight = torch.rand(1)
            image2_weight = 1 - image1_weight
            z_curr1 = _generate_noise_for_sampling(m, n, nzx, nzy, opt, NoiseMode.Z1) * image1_weight
            z_curr2 = _generate_noise_for_sampling(m, n, nzx, nzy, opt, NoiseMode.Z2) * image2_weight
            z_curr = functions.merge_noise_vectors(z_curr1, z_curr2, opt.noise_vectors_merge_method)

            if images_prev == []:
                I_prev = m(in_s)
                #I_prev = m(I_prev)
                #I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                I_prev = I_prev[:, :, 0:round(scale_v * reals1[n].shape[2]), 0:round(scale_h * reals1[n].shape[3])]
                I_prev = m(I_prev)
                I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])


            if n < gen_start_scale:
                z_curr = torch.cat((Z_opt1, Z_opt2))

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)

            if n == len(reals1)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
                    print(f"Saving image: {i}")
                    plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
                    #plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                    #plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)
            images_cur.append(I_curr)
        print(f"Done Generating level: {n}")
        n+=1
    return I_curr.detach()


def _generate_noise_for_sampling(m, n, nzx, nzy, opt, noise_mode):
    if n == 0:
        z_curr = functions.generate_noise([1, nzx, nzy], device=opt.device, noise_mode=noise_mode,
                                          gaussian_noise_z_distance=opt.gaussian_noise_z_distance)
        z_curr = z_curr.expand(1, 3, z_curr.shape[2], z_curr.shape[3])
        z_curr = m(z_curr)
    else:
        z_curr = functions.generate_noise([opt.nc_z, nzx, nzy], device=opt.device, noise_mode=noise_mode,
                                          gaussian_noise_z_distance=opt.gaussian_noise_z_distance)
        z_curr = m(z_curr)
    return z_curr

