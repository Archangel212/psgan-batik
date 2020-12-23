import torch
import torch.nn as nn
from config import opt
from torchvision import models

norma = nn.BatchNorm2d

def calc_gradient_penalty(netD, real_data, fake_data):
    from torch import autograd
    LAMBDA=1
    BATCH_SIZE=fake_data.shape[0]
    alpha = torch.rand(BATCH_SIZE).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    device=real_data.get_device()
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

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
            if i==nDepG-1:
                nf=1
            else:
                nf = ndf*2**i
            layers+=[nn.Conv2d(of, nf, 5, 2, 2)]##needs input 161 #hmm, also worls loke this
            if i !=0 and i !=nDepG-1:
                if True:#not opt.WGAN:
                    layers+=[norma(nf )]

            if i < nDepG -1:
                layers+=[nn.LeakyReLU(0.2, inplace=True)]
            else:
                if bSigm:
                    layers+=[nn.Sigmoid()]
            of = nf
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        if opt.WGAN:
            return output.mean(3).mean(2).unsqueeze(2).unsqueeze(3)
        return output#[:,:,1:-1,1:-1]

##################################################
class NetG(nn.Module):
    # @param ngf is channels of first layer, doubled up after every stride operation, or halved after upsampling
    # @param nDepG is depth, both of decoder and of encoder
    # @param nz is dimensionality of stochastic noise we add
    def __init__(self, ngf, nDepG, nz,nc=3):
        super(NetG, self).__init__()

        of = nz
        layers = []
        for i in range(nDepG):

            if i == nDepG - 1:
                nf = nc
            else:
                nf = ngf * 2 ** (nDepG - 2 - i)
            for j in range(opt.nBlocks):
                layers += [ResnetBlock(of, padding_type="zero", norm_layer=norma, use_dropout=False, use_bias=True)]

            layers += [nn.Upsample(scale_factor=2, mode='nearest')]  # nearest is default anyway
            layers += [nn.Conv2d(of, nf, 5, 1, 2)]
            if i == nDepG - 1:
                layers += [nn.Tanh()]
            else:
                layers += [norma(nf)]
                layers += [nn.ReLU(True)]
            of = nf
        self.G = nn.Sequential(*layers)

    def forward(self, input):
        return self.G(input)

class NetUskip(nn.Module):
    # @param nc is output channels
    # @param ncIn is input channels
    # @param ngf is channels of first layer, doubled up after every stride operation, or halved after upsampling
    # @param nDepG is depth, both of decoder and of encoder
    # @param nz is dimensionality of stochastic noise we add
    # @param NCbot can optionally specify the bottleneck size explicitly
    # @param bCopyIn copies the input channels at every decoder level -- special skip connections
    # @param bSkip turns skip connections btw encoder and decoder off
    # @param bTanh turns nonlinearity on and off
   def __init__(self, ngf, nDepG, nz=0, Ubottleneck=-1, nc=3, ncIn=None, bSkip=True, bTanh =True, bCopyIn=False):
        super(NetUskip, self).__init__()
        self.nDepG = nDepG
        self.eblocks=nn.ModuleList()
        self.dblocks=nn.ModuleList()
        self.bSkip=bSkip
        self.bCopyIn = bCopyIn

        if Ubottleneck <=0:
            Ubottleneck= ngf * 2 ** (nDepG - 1)

        if ncIn is None:	        
            of = nc
        else:
            of=ncIn ##in some cases not an RGB conditioning

        of += opt.first_noise*nz
        for i in range(self.nDepG):
            layers = []
            if i == self.nDepG-1:
                nf = Ubottleneck
            else:
                nf = ngf*2**i
            if i >0 and self.bCopyIn:##add coordinates
                of +=2#ncIn
        
            layers+=[nn.Conv2d(of, nf, 5, 2, 2)]
            if i !=0:
                layers+=[norma(nf )]
            if i < self.nDepG -1:
                layers+=[nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers+=[nn.Tanh()]
            of = nf
            block= nn.Sequential(*layers)
            self.eblocks +=[block]

        ##first nDepG layers
        of = nz + Ubottleneck - opt.first_noise * nz
        for i in range(nDepG):
            layers = []
            if i==nDepG-1:
                nf=nc
            else:
                nf = ngf*2**(nDepG-2-i)
            
            if i>0 and self.bSkip:
               of*=2##the u skip connections
               if self.bCopyIn:
                    of +=2

            for j in range(opt.nBlocks):
                layers += [ResnetBlock(of, padding_type="zero", norm_layer=norma, use_dropout=False, use_bias=True)]

            layers +=[nn.Upsample(scale_factor=2, mode='nearest')]#nearest is default anyway
            layers+=[nn.Conv2d(of,nf, 5, 1, 2)]
            if i == nDepG-1:
                if bTanh:
                    layers+=[nn.Tanh()]
            else:
                layers+=[norma(nf)]
                layers+=[nn.ReLU(True)]
            of = nf
            block = nn.Sequential(*layers)
            self.dblocks +=[block]


   def forward(self, input1,input2=None):
        if opt.first_noise and input2 is not None:
            x = torch.cat([input1,nn.functional.upsample(input2,scale_factor=2**self.nDepG,mode='bilinear')],1)
            input2=None
        else:
            x=input1##initial input

        skips = []
        input1 = input1[:, 3:5]##only coords
        for i in range(self.nDepG):
            if i >0 and self.bCopyIn:
                input1=nn.functional.avg_pool2d(input1,int(2))
                x=torch.cat([x,input1],1)
            x= self.eblocks[i].forward(x)
            if i !=self.nDepG-1:
                if self.bCopyIn:
                    skips += [torch.cat([x,nn.functional.avg_pool2d(input1,int(2))],1)]
                else:
                    skips +=[x]
        bottle=x
        if input2 is not None:
            bottle = torch.cat((x,input2),1)##the det. output and the noise appended       
        x=bottle
        for i in range(len(self.dblocks)):
            x= self.dblocks[i].forward(x) 
            if i< self.nDepG-1 and self.bSkip:
                x = torch.cat((x,skips[-1-i]),1)
        return x


KER=5
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
