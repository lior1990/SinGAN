import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, add_batch_norm=True):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        if add_batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1, add_batch_norm=opt.add_batch_norm)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1, add_batch_norm=opt.add_batch_norm)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt, img_shape):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1, add_batch_norm=opt.add_batch_norm)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1, add_batch_norm=opt.add_batch_norm)
            self.body.add_module('block%d'%(i+1),block)
        self.img_output_channels = opt.nc_im
        output_channels = self.img_output_channels
        if opt.enable_mask:
            output_channels += 1  # for mask
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),output_channels,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )

        if opt.enable_mask:
            if opt.mask_activation_fn == "sigmoid":
                self.mask_activation_layer = nn.Sequential(nn.Sigmoid())
            elif opt.mask_activation_fn == "relu_sign":
                self.mask_activation_layer = nn.Sequential(nn.ReLU(), Sign())
            elif opt.mask_activation_fn == "tanh_relu":
                self.mask_activation_layer = nn.Sequential(nn.Tanh(), nn.ReLU())
            elif opt.mask_activation_fn == "tanh_sign":
                self.mask_activation_layer = nn.Sequential(nn.Tanh(), Sign())
            elif opt.mask_activation_fn == "down_up":
                self.mask_activation_layer = nn.Sequential(nn.Tanh(), nn.ReLU(), nn.MaxPool2d(opt.ker_size), nn.Upsample(size=img_shape[2:]))
            else:
                raise NotImplementedError
        else:
            self.mask_activation_layer = None

    def forward(self, noise, prev):
        noise = self.head(noise)
        noise = self.body(noise)
        noise = self.tail(noise)
        output_img = noise[:, 0:self.img_output_channels, :, :]

        ind = int((prev.shape[2]-noise.shape[2])/2)
        prev = prev[:,:,ind:(prev.shape[2]-ind),ind:(prev.shape[3]-ind)]

        final_output_img = output_img + prev

        if self.mask_activation_layer is not None:
            mask = self.mask_activation_layer(noise[:, self.img_output_channels:self.img_output_channels+1, :, :])
            mask1 = mask
            mask2 = 1 - mask1
            mask1_output = mask1 * final_output_img
            mask2_output = mask2 * final_output_img
            return final_output_img, mask1_output, mask2_output
        else:
            return (final_output_img,)


class Sign(nn.Module):
    def forward(self, x):
        return torch.sign(x)

