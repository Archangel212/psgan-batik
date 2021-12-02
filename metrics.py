import os
import torch
from PIL import Image
from torch._C import device
from torchvision import transforms
import numpy as np
from prdc import compute_prdc
from pathlib import Path

from network import NetG
from utils import setNoise
from config import opt
from torchvision import utils as vutils
import torchvision.models as models

from config import opt


class FeatureExtractorPytorch:
    def __init__(self):
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',pretrained=True)
        self.model = models.vgg16(pretrained=True)

        self.model.eval()

        self.model.classifier = self.model.classifier[:-1]

    def preprocess_input(self, img):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        return input_batch

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        
        input_batch = self.preprocess_input(img) 
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            feature = self.model(input_batch)[0]  # (1, 4096) -> (4096, )
            feature = feature.cpu().detach().numpy()
        return feature / np.linalg.norm(feature)  # Normalize


fe = FeatureExtractorPytorch()

#real features
# dataset_path = Path("batik_dataset/32_Kawung")
dataset_path = Path(opt.texture_path)
real_features_path = Path("inference") / Path(dataset_path.name + "_features")
def construct_real_features(dataset_path):
    real_features = np.array([
        fe.extract(Image.open(batik)) for batik in dataset_path.glob("*.jpg")
    ])

    np.save(real_features_path, real_features)
    return real_features

real_features = construct_real_features(dataset_path)
# real_features = np.load(real_features_path.as_posix() + ".npy", allow_pickle=True)


# model_path = Path("/Users/apple/Downloads/log/32_Kawung/kernel=5,zl_dim=20,zg_dim=40,zp_dim=4,ngf=64,ndf=64,batch_size=25/MLP,samefakeimg,G_upsampleConv2d,instance_noise_mean=0.1,shuffle_ds=False,real_label_smoothing=1,nBlocksG=2,nBlocksG_padding=reflect/generator_model_e100.pth")

model_path = Path(opt.output_folder) / "generator_model_e100.pth"

fake_images_path = Path("inference") / Path(*list(model_path.parts[1:-1]))
os.makedirs(fake_images_path, exist_ok=True)
 
#generate fake images from saved generator model
def generate_fake_images():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    netG = NetG(int(opt.ngf), opt.nDepG, opt.nz).to(device)
    model = torch.load(model_path, map_location=device)

    netG.load_state_dict(model)

    netG.eval()

    noise = torch.FloatTensor(opt.batch_size, opt.nz, opt.NZ, opt.NZ)
    noise = noise.to(device)

    noise = setNoise(noise)
    fake = netG(noise)

    for idx, f in enumerate(fake):
        vutils.save_image(f, fake_images_path / "fake_{}.jpg".format(idx), normalize=True)


    np.save(fake_images_path / "fake_images.npy", fake.cpu().detach().numpy())
    return fake

fake = generate_fake_images()
# fake = np.load(fake_images_path / "fake_images.npy", allow_pickle=True)

def construct_fake_features():
    fake_features = np.array([
        fe.extract(Image.open(f)) for f in fake_images_path.glob("*.jpg")
    ])

    np.save(fake_images_path / "fake_features.npy", fake_features)

    return fake_features

fake_features = construct_fake_features()
# fake_features = np.load(fake_images_path / "fake_features.npy", allow_pickle=True)


#size=[num_real_samples, feature_dim]
metrics = compute_prdc(real_features=real_features,
                       fake_features=fake_features,
                       nearest_k=5)

print("density: {}, coverage: {}".format(metrics["density"], metrics["coverage"]))
