import argparse
import torch.nn as nn
import datetime
import os
parser = argparse.ArgumentParser()

##data path and loading parameters
parser.add_argument('--texture_path', required=True, help='path to texture image folder')
parser.add_argument('--content_path', default='', help='path to content image folder')
parser.add_argument('--mirror', type=bool, default=False,help='augment style image distribution for mirroring')
parser.add_argument('--content_scale', type=float, default=1.0,help='scale content images')
parser.add_argument('--texture_scale', type=float, default=1.0,help='scale texture images')
parser.add_argument('--test_image',default='None', help='path to test image file')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)#0 means a single main process
parser.add_argument('--output_folder', default='.', help='folder to output images and model checkpoints')
##neural network parameters
parser.add_argument('--batch_size', type=int, default=25, help='input batch size')
parser.add_argument('--image_size', type=int, default=160, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64,help='number of channels of generator (at largest spatial resolution)')
parser.add_argument('--ndf', type=int, default=64,help='number of channels of discriminator (at largest spatial resolution)')
parser.add_argument('--nDepG', type=int, default=5,help='depth of Unet Generator')
parser.add_argument('--nDepD', type=int, default=5,help='depth of DiscrimblendMoinator')
parser.add_argument('--N', type=int, default=30,help='count of memory templates')
parser.add_argument('--coord_copy', type=bool, default=True,help='copy  x,y coordinates of cropped memory template')
parser.add_argument('--multi_scale', type=bool, default=False,help='multi-scales of mixing features; if False only full resolution; if True all levels')
parser.add_argument('--nBlocks', type=int, default=0,help='additional res blocks for complexity in the unet')
parser.add_argument('--blend_mode', type=int, default=0,help='type of blending for parametric/nonparametric output')
parser.add_argument('--refine', type=bool, default=False,help='second unet after initial templates')
parser.add_argument('--skip_connections', type=bool, default=True,help='skip connections in  Unet -- allows better content reconstruct')
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
parser.add_argument('--WGAN', type=bool, default=False,help='use WGAN-GP adversarial loss')
parser.add_argument('--LS', type=bool, default=False,help='use least squares GAN adversarial loss')
##Optimisation parametersfalp
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dIter', type=int, default=1, help='number of Discriminator steps -- for 1 Generator step')
##noise parameters
parser.add_argument('--zLoc', type=int, default=40,help='noise channels, sampled on each spatial position')
parser.add_argument('--zGL', type=int, default=20,help='noise channels, identical on every spatial position')
parser.add_argument('--zPeriodic', type=int, default=3,help='periodic spatial waves')
parser.add_argument('--first_noise', type=bool, default=False,help='stochastic noise at bottleneck or input of Unet')
opt = parser.parse_args()

nDepG = opt.nDepG
##noise added to the deterministic content mosaic modules -- in some cases it makes a difference, other times can be ignored
bfirst_noise=opt.first_noise
nz=opt.zGL+opt.zLoc+opt.zPeriodic
bMirror=opt.mirror##make for a richer distribution, 4x times more data
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

if opt.output_folder=='.':
    i = opt.texture_path[:-1].rfind('/')
    i2 = opt.content_path[:-1].rfind('/')
    opt.output_folder = "results/"+opt.texture_path[i+1:]+opt.content_path[i2+1:]##actually 2 nested folders -- cool
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    opt.output_folder += stamp + "/"




