import os
from pathlib import Path

import torch
from torchvision.transforms.functional import normalize
from config import opt
from PIL import Image
import torchvision as vutils
import matplotlib.pyplot as plt

# count = 0
# for roots, dirs, files in os.walk("log"):
#   for f in files:
#     if f == "options.txt":
#       f = os.path.join(roots, f)
#       with open(f, "r") as file:
#         if "mirror=True" in file.read():
#           print(f,'\n')
#           count += 1

#   # break
# print(count)


from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from PIL import Image
import numpy as np


def augment_images(images,images_name,quantity=16,
                   path_to_save='Augmented_Batik_Images_500by500'):

  datagen = ImageDataGenerator(
    rotation_range=90,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True
  )
  for img,img_name in zip(images,images_name):
      for i in range(quantity):
          augmented_image = datagen.random_transform(img)
          pil_image = Image.fromarray(np.uint8(augmented_image))
          
          path = os.path.join(path_to_save,
                              img_name 
                              + "_transformation_" 
                              + str(i+1) 
                              + ".jpg")
          pil_image.save(path,'JPEG',quality=90)


def crop(image):
  cropped_images = []

  left = 2
  up = 2
  right = 162
  down = 162

  #height
  for _ in range(4):
    #width
    for _ in range(8): #24 times cropping

      cropped_image = np.array(image.crop((left, up, right, down)))
      # plt.imshow(cropped_image)
      # plt.show()

      left += 162
      right += 162

      cropped_images.append(cropped_image)

    up += 162
    down += 162

    left = 2
    right = 162

  return np.array(cropped_images)

    
def rearrange_generated_image():

  cwd = "/Users/apple/Documents/skripsi/psgan-batik/log"

  datasets = [
    "32_Kawung", "32_Parang", "32_Nitik", "32_Lereng", "32_Ceplok",
    "64_Kawung", "64_Parang", "64_Nitik", "64_Lereng", "64_Ceplok",
    "128_Kawung", "128_Parang", "128_Nitik", "128_Lereng", "128_Ceplok"
  ]

  model_hyperparameters = "kernel=5,zl_dim=20,zg_dim=40,zp_dim=4,ngf=64,ndf=64,batch_size=25"
  model_state = "MLP,samefakeimg,G_upsampleConv2d,instance_noise_mean=0.1,shuffle_ds=False,real_label_smoothing=1,nBlocksG=2,nBlocksG_padding=reflect"


  file = "generated_textures_099.jpg"
  target_file = "cleaned_generated_textures_099.jpg"

  for dataset in datasets:
    file_name = Path(cwd) / dataset / model_hyperparameters / model_state / file
    target_name = Path(cwd) / dataset / model_hyperparameters / model_state / target_file
    image = Image.open(file_name)

    #cropping
    cropped = crop(image)[:25] #take only 25 images, and remove the 7 black images 
    cropped = np.moveaxis(cropped, [1,2], [2,3])
    image_tensor = torch.from_numpy(cropped).type("torch.FloatTensor")
    print(file_name, image_tensor.size(), image_tensor.type())
    vutils.utils.save_image(image_tensor, target_name, normalize=True, nrow=5)
    

rearrange_generated_image()