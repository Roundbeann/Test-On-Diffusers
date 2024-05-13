
import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt

from diffusers import StableDiffusionPipeline

device = "cuda"

# img_path = f"./dog.png"
# mask_path = f"./dog_mask.png"
#
# img = Image.open(img_path)
# mask = Image.open(mask_path)

# Load the pipeline
model_id = "/data2/yuanshou/tmp/sdwebui/to3D/310/stable-dreamfusion-main/model/v2-1/v2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16).to(device)

generator = torch.Generator(device=device).manual_seed(42)

import torchvision.transforms as transforms
import cv2 as cv

img = cv.imread('/data2/yuanshou/tmp/data/pics/android.jpg')
print(img.shape)  # numpy数组格式为（H,W,C）

transf = transforms.ToTensor()
img_tensor = transf(img).view(1, 3, 512, 512).to(device)  # tensor数据格式是torch(C,H,W)

print(img_tensor.shape)

# Create some fake data (a random image, range (-1, 1))
images = torch.rand(1, 3, 512, 512).to(device) * 2 - 1  # （-1,1）
images = images.to(torch.float16)
print(images.shape)

# Encode to latent space
with torch.no_grad():
    latents = 0.18215 * pipe.vae.encode(img_tensor).latent_dist.mean  # 0.18215 scaling factor
print("Encoded latents shape:", latents.shape)
