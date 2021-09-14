import argparse
import torch.nn as nn
import datetime
import os

parser = argparse.ArgumentParser()

##data path and loading parameters
parser.add_argument('--texture_path', default='', help='path to texture image folder')
parser.add_argument('--content_path', default='', help='path to content image folder')
parser.add_argument('--mirror', type=bool, default=True,help='augment style image distribution for mirroring')
parser.add_argument('--texture_scale', type=float, default=1.0,help='scale texture images')
parser.add_argument('--content_scale', type=float, default=1.0,help='scale content images')
parser.add_argument('--test_image',default='None', help='path to test image file')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)#0 means a single main process
parser.add_argument('--output_folder', default='.', help='folder to output images and model checkpoints')
##neural network parameters
parser.add_argument('--kernel_size', type=int, default=5, help='size of kernel, default 5')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--image_size', type=int, default=160, help='the height / width of the input image to network')
parser.add_argument('--shuffle_ds', type=bool, default=False, help='shuffle dataset, default is True')
parser.add_argument('--ngf', type=int, default=128,help='number of channels of generator (at largest spatial resolution)')
parser.add_argument('--ndf', type=int, default=128,help='number of channels of discriminator (at largest spatial resolution)')
parser.add_argument('--nDepG', type=int, default=5,help='depth of Unet Generator')
parser.add_argument('--nDepD', type=int, default=5,help='depth of Discriminator')
parser.add_argument('--N', type=int, default=30,help='count of memory templates')
parser.add_argument('--BN_D', type=bool, default=True,help='Batch Norm of Discriminator, default is True')
parser.add_argument('--nBlocks', type=int, default=0,help='additional res blocks for complexity in the unet')
parser.add_argument('--Ubottleneck', type=int, default=-1,help='Unet bottleneck, leave negative for default wide bottleneck')
##regularization and loss criteria weighting parameters
parser.add_argument('--fContent', type=float, default=1.0,help='weight of content reconstruction loss')
parser.add_argument('--fAdvM', type=float, default=.0,help='weight of I_M adversarial loss')
parser.add_argument('--fContentM', type=float, default=1.0,help='weight of I_M content reconstruction loss')
parser.add_argument('--cLoss', type=int, default=0,help='type of perceptual distance metric for reconstruction loss')
parser.add_argument('--fAlpha', type=float, default=.1,help='regularization weight of norm of blending mask')
parser.add_argument('--fTV', type=float, default=.1,help='regularization weight of total variation of blending mask')
parser.add_argument('--fEntropy', type=float, default=.5,help='regularization weight of entropy -- forcing low entropy results in 0/1 values in mix tensor A')
parser.add_argument('--fDiversity', type=float, default=1,help='regularization weight of diversity of used templates')
##Optimisation parameters
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for Generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for Discriminator, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam default 0')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dIter', type=int, default=1, help='number of Discriminator steps -- for 1 Generator step')
##set to true if wanna use WGAN
parser.add_argument('--WGAN', type=bool, default=False,help='use WGAN-GP adversarial loss')
parser.add_argument('--LS', type=bool, default=False,help='use least squares GAN adversarial loss')
##noise parameters
parser.add_argument('--zLoc', type=int, default=40,help='noise channels, sampled on each spatial position')
parser.add_argument('--zGL', type=int, default=20,help='noise channels, identical on every spatial position')
parser.add_argument('--zPeriodic', type=int, default=3,help='periodic spatial waves')
parser.add_argument('--first_noise', type=bool, default=False,help='stochastic noise at bottleneck or input of Unet')
##GAN Hacks by Soumith Chintala
parser.add_argument('--std_instance_noise', type=float, default=0.1,help='standard deviation of instance noise for real images default 0.1')
parser.add_argument('--real_label_smoothing', type=float, default=1,help='smooth positive labels percentage, default 0')
opt = parser.parse_args()

nz = opt.zGL + opt.zLoc + opt.zPeriodic
NZ = opt.image_size//2**opt.nDepG
opt.nz = nz
opt.NZ = NZ

opt.fContentM *= opt.fContent

##GAN criteria changes given loss options LS or WGAN
if not opt.WGAN and not opt.LS:
    criterion = nn.BCELoss()
elif opt.LS:
    def crit(x,l):
        return ((x-l)**2).mean()
    criterion=crit
else:
    def dummy(val,label):
        return (val*(1-2*label)).mean()#so -1 fpr real. 1 fpr fake
    criterion=dummy




