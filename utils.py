import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import PIL
import torch.nn as nn
from config import opt

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
import math

class TextureDataset(Dataset):
    """Dataset wrapping images from a random folder with textures

    Arguments:
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, img_path, transform=None,scale=1):
        self.img_path = img_path
        self.transform = transform    
        if True:##ok this is for 1 worker only!
            names = os.listdir(img_path)
            self.X_train =[]
            for n in names:
                name = os.path.join(self.img_path, n)
                try:
                    img = Image.open(name)
                    try:
                        img = img.convert('RGB')##fixes truncation???
                    except:
                        pass
                    if scale!=1:
                        img=img.resize((int(img.size[0]*scale),int(img.size[1]*scale)),PIL.Image.LANCZOS)
                except Exception as e:
                    print (e,name)
                    continue

                self.X_train +=[img]
                print (n,"img added", img.size,"total length",len(self.X_train))
                if len(self.X_train) > 4000:
                    break ##usually want to avoid so many files

        ##this affects epoch length..
        if len(self.X_train) < 2000:
            c = int(2000/len(self.X_train))
            self.X_train*=c

    def __getitem__(self, index):
        if False:
            name =self.img_path + self.X_train[index]
            img = Image.open(name)
        else:
            img= self.X_train[index]#np.random.randint(len(self.X_train))   
        if self.transform is not None:
            img2 = self.transform(img)        
        label =0
        #print ('data returned',img2.data.shape)
        return img2, label

    def __len__(self):
        return len(self.X_train)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if opt.zPeriodic:
  # 2*nPeriodic initial spread values
  # slowest wave 0.5 pi-- full cycle after 4 steps in noise tensor
  # fastest wave 1.5pi step -- full cycle in 0.66 steps
  def initWave(nPeriodic):
    buf = []
    for i in range(nPeriodic // 4+1):
        v = 0.5 + i / float(nPeriodic//4+1e-10)
        buf += [0, v, v, 0]
        buf += [0, -v, v, 0]  # #so from other quadrants as well..
    buf = buf[:2*nPeriodic]
    awave = np.array(buf, dtype=np.float32) * np.pi
    awave = torch.FloatTensor(awave).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    return awave
  waveNumbers = initWave(opt.zPeriodic).to(device)

  class Waver(nn.Module):
    
    def __init__(self, input_size=(25, 20, 5, 5)):
      super(Waver, self).__init__()
      if opt.zGL > 0:
        K = 120
        batch_size, zGl, NZ, NZ = input_size
        layers = [nn.Flatten(start_dim=0, end_dim=-1)]
        layers += [nn.Linear(batch_size * zGl * NZ * NZ, K)]
        layers += [nn.ReLU(True)]
        layers += [nn.Linear(K, batch_size * 2 * opt.zPeriodic * NZ * NZ)]
        self.learnedWN =  nn.Sequential(*layers)
      else:##static
        self.learnedWN = nn.Parameter(torch.zeros(opt.zPeriodic * 2).uniform_(-1, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * 0.2)

    def forward(self, c, zGL=None):
      if opt.zGL > 0:
        #c shape: (batch_size, 2*opt.zPeriodic, NZ, NZ)
        #waveNumbers shape: (1, 2*opt.zPeriodic, 1, 1)
        #self.learnedWN(zGL) output shape : (batch_size, 2*opt.zPeriodic, NZ, NZ)
        #returned shape will be : (batch_size, 2*opt.zPeriodic, NZ, NZ)
        learned_wavenumbers = self.learnedWN(zGL).view(opt.batch_size, 2*opt.zPeriodic, opt.NZ, opt.NZ)

        return (waveNumbers + 5*learned_wavenumbers) * c

      return (waveNumbers + self.learnedWN) * c
  learnedWN = Waver(input_size=(opt.batch_size, opt.zGL, opt.NZ, opt.NZ))
else:
  learnedWN = None

##inplace set noise
def setNoise(noise):
  noise = noise.detach() * 1.0
  noise.uniform_(-1, 1)  # normal_(0, 1)
  if opt.zGL:
    noise[:, :opt.zGL] = noise[:, :opt.zGL, :1, :1].repeat(1, 1, noise.shape[2], noise.shape[3])
  if opt.zPeriodic:
    xv, yv = np.meshgrid(np.arange(noise.shape[2]), np.arange(noise.shape[3]),indexing='ij')
    c = torch.FloatTensor(np.concatenate([xv[np.newaxis], yv[np.newaxis]], 0)[np.newaxis])
    c = c.repeat(noise.shape[0], opt.zPeriodic, 1, 1)
    c = c.to(device)
    # #now c has canonic coordinate system -- multiply by wave numbers
    raw = learnedWN(c, noise[:, :opt.zGL])
    #random phase offset , it mimics random positional extraction of patches from the real images
    offset = (noise[:, -opt.zPeriodic:, :1, :1] * 1.0).uniform_(-1, 1) * 6.28
    offset = offset.repeat(1, 1, noise.shape[2], noise.shape[3])
    wave = torch.sin(raw[:, ::2] + raw[:, 1::2] + offset)
    noise[:, -opt.zPeriodic:] = wave
  return noise

def save_model(epoch, generator, generator_optimizer, discriminator, discriminator_optimizer, output_folder):
  # saving training result
  # generator.save(add_state={'optimizer_state_dict' : generator_optimizer.state_dict()},
  #             file_name=os.path.join(output_folder,'generator_param_fin_{}.pth'.format(epoch+1, datetime.now().strftime("%Y%m%d_%H-%M-%S"))))
  # discriminator.save(add_state={'optimizer_state_dict' : discriminator_optimizer.state_dict()},
  #             file_name=os.path.join(output_folder,'discriminator_param_fin_{}.pth'.format(epoch+1, datetime.now().strftime("%Y%m%d_%H-%M-%S"))))

  model_output_path = os.path.join(output_folder,"generator_model_e{}.pth".format(epoch))
  optimizer_output_path = os.path.join(output_folder,"generator_optimizer_e{}.pth".format(epoch))

  torch.save(generator.state_dict(), model_output_path)

  torch.save(generator_optimizer.state_dict(), optimizer_output_path)
  

def plot_loss(log_dir):
  plt.figure(figsize=(5,5))

  #find on csv file
  csv_path = [p for p in Path(log_dir).rglob("*.csv")][0]
  df = pd.read_csv(csv_path,index_col=None)

  loss_path = os.path.join(log_dir, "plot")
  os.makedirs(loss_path, exist_ok=True)

  plt.subplot(2,1,1)
  plt.plot(df["epoch"], df["lossG"], color="r")
  plt.xlabel("epoch")
  plt.title("generator_loss")
  
  plt.subplot(2,1,2)
  plt.plot(df["epoch"], df["lossD"], color="b")
  plt.xlabel("epoch")
  plt.title("discriminator_loss_real&fakeimg")
  
  plt.tight_layout(pad=2.0)
  plt.savefig(os.path.join(loss_path,"loss.jpg"))
  plt.close()

  #============================================================

  plt.subplot(3,1,1)
  plt.plot(df["epoch"], df["D_x"], color="r")
  plt.xlabel("epoch")
  plt.title("D_output_on_realimgs")

  plt.subplot(3,1,2)
  plt.plot(df["epoch"], df["D_G_z1"], color="r")
  plt.xlabel("epoch")
  plt.title("D_output_on_fakeimgs_fakelabel")

  plt.subplot(3,1,3)
  plt.plot(df["epoch"], df["D_G_z2"], color="r")
  plt.xlabel("epoch")
  plt.title("D_output_on_fakeimgs_reallabel")
  
  plt.tight_layout(h_pad=1.0)
  plt.savefig(os.path.join(loss_path,"D_output.jpg"))
  plt.close()
  
def smooth_real_labels(y, percentage=0.382):
    
  #randomize the label into range 0.7 to 1.2 according to GAN Hacks by S.Chintala
  unraveled_y = y.view(-1)
  len_unraveled_y = len(unraveled_y)
  amount = math.ceil(len_unraveled_y*percentage)
  random_idx = random.sample(range(len_unraveled_y), amount)

  for i in range(len(unraveled_y)):
    if i in random_idx:
      unraveled_y[i] = 1 - 0.3 + (random.random() * 0.5)

  return unraveled_y.view(y.shape)