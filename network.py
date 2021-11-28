import torch
import torch.nn as nn
from config import opt
from torchvision import models
from config import opt

norma = nn.BatchNorm2d

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator(nn.Module):
    # @param ncIn is input channels
    # @param ndf is channels of first layer, doubled up after every conv. layer with stride
    # @param nDepG is depth
    # @param bSigm is final nonlinearity, off for some losses
    def __init__(self, ndf, nDepG, ncIn=3, bSigm=True):
        super(Discriminator, self).__init__()

        layers = []
        of = ncIn
        for i in range(nDepG):
            if i == nDepG-1:
                nf = 1
            else:
                nf = ndf*2**i

            for j in range(opt.nBlocksD):
                layers += [ResnetBlock(of, padding_type="zero", norm_layer=norma, use_dropout=True, use_bias=True) ]
                
            layers += [nn.Conv2d(of, nf, opt.kernel_size, 2, 2)]##needs input 161 #hmm, also worls loke this

            if i != 0 and i != nDepG-1:
                if opt.BN_D:
                    layers += [norma(nf)]

            if i < nDepG -1:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
                # layers += [nn.Dropout(p=0.2)]
            else:
                if bSigm:
                    layers += [nn.Sigmoid()]
            of = nf
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        return output#[:,:,1:-1,1:-1]

##################################################
class NetG(nn.Module):
    # @param ngf is channels of first layer, doubled up after every stride operation, or halved after upsampling
    # @param nDepG is depth, both of decoder and of encoder
    # @param nz is dimensionality of stochastic noise we add
    def __init__(self, ngf, nDepG, nz, nc=3):
        super(NetG, self).__init__()

        of = nz
        layers = []
        for i in range(nDepG):

            if i == nDepG - 1:
                nf = nc
            else:
                nf = ngf * 2 ** (nDepG - 2 - i)
            for j in range(opt.nBlocksG):
                layers += [ResnetBlock(of, padding_type="zero", norm_layer=norma, use_dropout=False, use_bias=True)]

            layers += [nn.Upsample(scale_factor=2, mode='nearest')]  # nearest is default anyway
            layers += [nn.Conv2d(of, nf, opt.kernel_size, 1, 2)]
            if i == nDepG - 1:
                layers += [nn.Tanh()]
            else:
                layers += [norma(nf)]
                layers += [nn.ReLU(inplace=True)]
            of = nf
        self.G = nn.Sequential(*layers)

    def forward(self, input):
        return self.G(input)

KER=opt.kernel_size
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.gamma = nn.Parameter(torch.zeros(1))

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(KER//2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(KER//2)]
        elif padding_type == 'zero':
            p = KER//2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=KER, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(KER//2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(KER//2)]
        elif padding_type == 'zero':
            p = KER//2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=KER, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.gamma*self.conv_block(x)
        return out
