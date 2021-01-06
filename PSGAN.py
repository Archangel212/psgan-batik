from __future__ import print_function
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from utils import TextureDataset, setNoise, learnedWN, save_model,plot_loss
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import sys
from network import weights_init, Discriminator, calc_gradient_penalty, NetG
from config import opt
import time
from train_logger import TrainLogger
import os


os.makedirs(opt.output_folder, exist_ok=True)
print("\nsaving at {}\n".format(opt.output_folder))

text_file = open(os.path.join(opt.output_folder,"options.txt"), "w")
text_file.write(str(opt))
text_file.close()
print (opt)

#=======================================================

if opt.manualSeed is None:
  opt.manualSeed = 618
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

canonicT = [transforms.RandomCrop(opt.image_size), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
mirrorT = []
if opt.mirror:
  mirrorT += [transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip()]
transformTex = transforms.Compose(mirrorT+canonicT)
dataset = TextureDataset(opt.texture_path, transformTex, opt.texture_scale)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers = int(opt.workers))

criterion = nn.BCELoss()

ngf = int(opt.ngf)
ndf = int(opt.ndf)

noise = torch.FloatTensor(opt.batch_size, opt.nz, opt.NZ, opt.NZ)
fixnoise = torch.FloatTensor(opt.batch_size, opt.nz, opt.NZ*4, opt.NZ*4)

netD = Discriminator(ndf, opt.nDepD)
##################################
netG = NetG(ngf, opt.nDepG, opt.nz)
print(NetG, netD)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("device",device)

Gnets=[netG]
if opt.zPeriodic:
  Gnets += [learnedWN]

for net in [netD] + Gnets:
  try:
    net.apply(weights_init)
  except Exception as e:
    print (e,"weightinit")
  pass
  net = net.to(device)
  print(net)


real_label = 1
fake_label = 0

noise = noise.to(device)
fixnoise = fixnoise.to(device)

# setup optimizer
optimizerG = optim.Adam([param for net in Gnets for param in list(net.parameters())], lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)#netD.parameters()

# for loggin the trainning
tlog = TrainLogger("train_log", log_dir=opt.output_folder, csv=True, header=True, suppress_err=False)
tlog.disable_pickle_object()
tlog.set_default_Keys(["epoch", "lossD", "lossG", "D_x", "D_G_z1", "D_G_z2"])

start = time.time()
for epoch in range(opt.niter):
  # for logging
  errD = 0.0
  errG = 0.0
  D_x = 0.0
  D_G_z1 = 0.0
  D_G_z2 = 0.0

  for i, data in enumerate(dataloader, 0):
    t0 = time.time()
    sys.stdout.flush()
    # train with real
    netD.zero_grad()
    textures, _ = data
    textures = textures.to(device)

    #apply instance noise  
    textures = textures + torch.normal(mean=0, std=opt.std_instance_noise, size=textures.size(), device=device)

    output = netD(textures)
    errD_real = criterion(output, output.detach()*0 + real_label)
    errD_real.backward()
    D_x = output.mean().item()

    # train with fake
    noise = setNoise(noise)
    fake = netG(noise)
    output = netD(fake.detach())
    errD_fake = criterion(output, output.detach()*0 + fake_label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()

    errD = errD_real + errD_fake
    if opt.WGAN:
      gradient_penalty = calc_gradient_penalty(netD, textures, fake[:textures.shape[0]])##for case fewer textures images
      gradient_penalty.backward()

    optimizerD.step()
    if i > 0 and opt.WGAN and i % opt.dIter != 0:
      continue ##critic steps to 1 GEN steps

    for net in Gnets:
      net.zero_grad()

    # noise = setNoise(noise)
    # fake = netG(noise)
    output = netD(fake)
    errG = criterion(output, output.detach()*0 + real_label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()

    print('[%d/%d][%d/%d]\tLossD: %.4f LossG: %.4f D(x): %.4f D(G(z)): %.4f / %.4f time %.4f'
          % (epoch, opt.niter, i, len(dataloader), 
          errD.item(), errG.item() ,D_x, D_G_z1, D_G_z2, time.time() - t0))

    if (epoch % 2 == 0 or epoch == (opt.niter - 1)) and (i == len(dataloader) - 1):
      vutils.save_image(textures, '%s/real_textures.jpg' % opt.output_folder,  normalize=True)
      vutils.save_image(fake, '%s/generated_textures_%03d.jpg' % (opt.output_folder, epoch), normalize=True)

      # fixnoise = setNoise(fixnoise)

      # vutils.save_image(fixnoise.view(-1,1,fixnoise.shape[2],fixnoise.shape[3]), '%s/noiseBig_epoch_%03d_%s.jpg' % (opt.output_folder, epoch), normalize=True)

      # netG.eval()
      # with torch.no_grad():
      #     fakeBig = netG(fixnoise)

      # vutils.save_image(fakeBig,'%s/big_texture_%03d_%s.jpg' % (opt.output_folder, epoch),normalize=True)
      # netG.train()

      ##OPTIONAL
      ##save/load model for later use if desired
      #outModelName = '%s/netG_epoch_%d_%s.pth' % (opt.output_folder, epoch*0)
      #torch.save(netU.state_dict(),outModelName )
      #netU.load_state_dict(torch.load(outModelName))

    elif (epoch % 100 == 0 or epoch == (opt.niter - 1)) and (i == len(dataloader) - 1):
      save_model(epoch, netG, optimizerG, netD, optimizerD, opt.output_folder)

  tlog.log([epoch+1, float(errD), float(errG), 
  float(D_x), float(D_G_z1), float(D_G_z2)])

save_model(epoch+1, netG, optimizerG, netD, optimizerD, opt.output_folder)
plot_loss(opt.output_folder)
elapsed_time = time.time() - start
print("Time for training: {} seconds".format(elapsed_time))

