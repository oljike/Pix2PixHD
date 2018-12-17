import os
import math
import glob
import random
import itertools
import numpy as np
from PIL import Image
from skimage import io
import dlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torch.utils.data
from torchvision import models
from torch.autograd import Variable
import torchvision.datasets as datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, generator):
        super(LocalEnhancer, self).__init__()

        model_global = generator
        #         model_global.conv7 = Identity()
        #         model_global.norm7 = Identity()
        #         model_global.act7  = Identity()
        model_global.pad8 = Identity()
        model_global.conv8 = Identity()
        model_global.act8 = Identity()

        self.model_global = model_global

        ###downsample
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(input_nc, 32, kernel_size=7, padding=0)
        self.norm1 = nn.InstanceNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(64)

        ### upsample
        num_bottlenecks = 3
        self.Bottleneck = nn.Sequential(*[
            ResnetBlock(64, nn.InstanceNorm2d(64)) for _ in range(num_bottlenecks)
        ])

        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm3 = nn.InstanceNorm2d(32)

        self.pad4 = nn.ReflectionPad2d(3)
        self.conv4 = nn.Conv2d(32, output_nc, kernel_size=7, padding=0)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input_):
        ### create input pyramid

        #         input_downsampled = [input]
        input_downsampled = input_.clone()
        input_downsampled = self.downsample(input_downsampled)

        output_global = self.model_global(input_downsampled)

        x_res = self.pad1(input_)
        x_res = self.conv1(x_res)
        x_res = self.norm1(x_res)
        x_res = F.relu(x_res)

        x_res = self.conv2(x_res)
        x_res = self.norm2(x_res)
        x_res = F.relu(x_res)

        x_out = self.Bottleneck(x_res + output_global)

        x_out = self.conv3(x_out)
        x_out = self.norm3(x_out)
        x_out = F.relu(x_out)

        x_out = self.pad4(x_out)
        x_out = self.conv4(x_out)
        x_out = torch.tanh(x_out)

        return x_out


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(GlobalGenerator, self).__init__()

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, padding=0)
        self.norm1 = nn.InstanceNorm2d(64)

        self.conv2 = nn.Conv2d(64 * 1, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(512)

        num_bottlenecks = 9
        self.Bottleneck = nn.Sequential(*[
            ResnetBlock(512, nn.InstanceNorm2d(512)) for _ in range(num_bottlenecks)
        ])

        self.conv5 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm5 = nn.InstanceNorm2d(256)

        self.conv6 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm6 = nn.InstanceNorm2d(128)

        self.conv7 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm7 = nn.InstanceNorm2d(64)
        self.act7 = nn.ReLU(True)

        self.pad8 = nn.ReflectionPad2d(3)
        self.conv8 = nn.Conv2d(64, output_nc, kernel_size=7, padding=0)
        self.act8 = nn.Tanh()

    def forward(self, input_):
        x = self.pad1(input_)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)

        x = self.Bottleneck(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.norm6(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = self.norm7(x)
        x = self.act7(x)

        x = self.pad8(x)
        x = self.conv8(x)
        x = self.act8(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer):
        super(ResnetBlock, self).__init__()

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=0)
        self.norm1 = norm_layer

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=0)
        self.norm2 = norm_layer

    def forward(self, x):
        x_res = self.pad1(x)
        x_res = self.conv1(x_res)
        x_res = self.norm1(x_res)
        x_res = F.relu(x_res)
        x_res = self.pad2(x_res)
        x_res = self.conv2(x_res)
        x_res = self.norm2(x_res)

        out = x + x_res
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=True, num_D=3, getIntermFeat=True):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
    # Defines the PatchGAN discriminator with the specified arguments.


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
