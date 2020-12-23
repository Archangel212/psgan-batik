import argparse
import torch.nn as nn
import datetime
import os

parser = argparse.ArgumentParser()

##data path and loading parameters
parser.add_argument('--texture_path', default='', help='path to texture image folder')
parser.add_argument('--mirror', type=bool, default=False,help='augment style image distribution for mirroring')
parser.add_argument('--texture_scale', type=float, default=1.0,help='scale texture images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)#0 means a single main process
parser.add_argument('--output_folder', default='.', help='folder to output images and model checkpoints')
##neural network parameters
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--image_size', type=int, default=160, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=120,help='number of channels of generator (at largest spatial resolution)')
parser.add_argument('--ndf', type=int, default=120,help='number of channels of discriminator (at largest spatial resolution)')
parser.add_argument('--nDepG', type=int, default=5,help='depth of Unet Generator')
parser.add_argument('--nDepD', type=int, default=5,help='depth of DiscrimblendMoinator')
parser.add_argument('--nBlocks', type=int, default=0,help='additional res blocks for complexity in the unet')
parser.add_argument('--Ubottleneck', type=int, default=-1,help='Unet bottleneck, leave negative for default wide bottleneck')
##Optimisation parametersfalp
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dIter', type=int, default=1, help='number of Discriminator steps -- for 1 Generator step')
##set to true if wanna use WGAN
parser.add_argument('--WGAN', type=bool, default=False,help='use WGAN-GP adversarial loss')
##noise parameters
parser.add_argument('--zLoc', type=int, default=50,help='noise channels, sampled on each spatial position')
parser.add_argument('--zGL', type=int, default=20,help='noise channels, identical on every spatial position')
parser.add_argument('--zPeriodic', type=int, default=0,help='periodic spatial waves')
parser.add_argument('--first_noise', type=bool, default=False,help='stochastic noise at bottleneck or input of Unet')
opt = parser.parse_args()







