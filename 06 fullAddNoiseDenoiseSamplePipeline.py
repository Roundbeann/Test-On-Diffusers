
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import trange

from diffusers import DDPMPipeline # https://huggingface.co/docs/diffusers/api/pipelines/ddpm

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training 

'''
Load in some images from the training data
Add noise, in different amounts.
Feed the noisy versions of the inputs into the model
Evaluate how well the model does at denoising these inputs
Use this information to update the model weights, and repeat
'''

## Step1: Downloading a training dataset

import torchvision
from datasets import load_dataset
from torchvision import transforms

dataset = load_dataset("/data2/yuanshou/tmp/sdwebui/models/butterflydata/butterflydataset", split="train") # imageFoder 
# git lfs clone https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset
image_size = 32 
batch_size = 64
len(dataset)

# dataset[0]["image"] # (512 w , 283 h)
# dataset[0]["image"].convert("RGB") # (512 w , 283 h)
dataset[0]["image"]

dataset[0]["image"].size

dataset[0]["image"].convert("RGB").size

# 数据增强操作
preprocess = transforms.Compose(
    [
        # 重设图片大小
        transforms.Resize((image_size, image_size)),  # Resize
        # 随机水平翻转图片
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        # 转化成tenser
        transforms.ToTensor(),  # Convert to tensor (0, 1) # b, c, h, w
        # 数据归一化
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

print("checkpoint1\n"+str(dataset))

print("checkpoint2\n"+str(dataset[0]["images"].shape)
      +"\ndatatype:\n"+str(type(dataset[0]["images"]))
      )

# 自动只提取了["images"]
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

# 数据增强结果展示
print("checkpoint3\n"+str(next(iter(train_dataloader)).keys()))
print("checkpoint4\n"+str(next(iter(train_dataloader))["images"].shape))

# grab a batch of images and visualize it
xb_visualize = next(iter(train_dataloader))["images"].to(device)[:64]   # torch.Size([64, 3, 32, 32])
plt.imshow(show_images(xb_visualize).resize((1 * 580, 580), resample=Image.BILINEAR))
plt.show()
## Step2: Define the Scheduler

from diffusers import DDPMScheduler

# 给定加噪和去噪的步长作为参数
noise_scheduler = DDPMScheduler()
noise_scheduler.config.num_train_timesteps=1000
print("noise_scheduler的介绍,默认线性分配噪声：\n",noise_scheduler)

# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.004)
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2') # better for small images
# 在图中插入公式
plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"ImageWeight${\sqrt{\bar{\alpha}_t}}$")
plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"NoiseWeight$\sqrt{(1 - \bar{\alpha}_t)}$")
plt.plot(noise_scheduler.alphas.cpu(), label=r"$\sqrt{ {\alpha}_t}$")

plt.legend(fontsize="x-large");
plt.show()

xb = xb_visualize

# 生成 64 个不同的扩散步长
timesteps = torch.linspace(0, 999, 64).long().to(device)
noise     = torch.randn_like(xb) # mean = 0 , std = 1


# noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# 调用noise_scheduler.add_noise，来对原图xb，分别在扩散步长timesteps上分配噪声noise
# 得到被分配噪声的噪声图像 noisy_xb
noisy_xb  = noise_scheduler.add_noise(xb, noise, timesteps) # xb: 64 
print("checkpoint5\nNoisy X shape", noisy_xb.shape)
print("checkpoint6\n"+str(timesteps))

# 展示加噪后的噪声图片
plt.imshow(show_images(noisy_xb).resize((1 * 640, 640), resample=Image.BILINEAR))
plt.show()
## Step3: Define Model UNet

# 从 diffusers模块 导入 Unet 网络
from diffusers import UNet2DModel

# 创建Unet网络
# create a model
model = UNet2DModel(
    sample_size = image_size, # target image resolution
    in_channels = 3,
    out_channels = 3,
    layers_per_block=2, # how many resnet layers to use per Unet block
    block_out_channels = (64, 128,128, 256),
    down_block_types   = (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",  # a regular ResNet upsampling block
    ),  
).to(device)

# with torch.no_grad():
#     model_prediction = model(noisy_xb, timesteps).sample
# model_prediction.shape  #  batchsize: 8 

# Step4: Create a Training Loop

import time
# 3090ti: Mem-Usage 3959MiB 16.1%    GPU-Utils: 37%   

# 定义新的噪声分配器，噪声的扩散步长为1000步 
# 噪声的分配不再采用线性方式，而是采用平方余弦的方式
# Set the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr = 4e-4)

losses = []

start_time = time.time()  # start timer at the beginning of each epoch

for epoch in trange(30):

    if epoch ==29:
        print()
    for step, batch in enumerate(train_dataloader):
        # 从train_dataloader得到训练图像
        clean_images = batch["images"].to(device)
        # 得到初始化的噪声
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        # plt.imshow(show_images(noise).resize((1 * 640, 640), resample=Image.BILINEAR))
        # plt.show()
        # 得到batch的大小
        bs = clean_images.shape[0] # 64
        # 得到给每张图片设定的加噪步长
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs, ), device = clean_images.device
        ).long()

        # 对清晰图像施加对应噪声步长的噪声，得到模糊图像/噪声图像
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        # plt.imshow(show_images(noisy_images).resize((1 * 640, 640), resample=Image.BILINEAR))
        # plt.show()
        # 使用Unet，根据噪声图片和噪声图片对应的加噪步长，预测初始噪声含量
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        # 注意，扩散模型预设【对每一张图像，每一个扩散步长，施加相同的噪声】
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)
        losses.append(loss.item())
        # plt.imshow(show_images(noise_pred).resize((1 * 640, 640), resample=Image.BILINEAR))
        # plt.title(f"batchEpoch = {epoch+1},loss={loss}")
        # plt.show()
        # 更新 Unet 参数
        optimizer.step()
        optimizer.zero_grad()
        # print(f"batchEpoch:{epoch+1}, loss: {loss}")

    plt.imshow(show_images(noisy_images).resize((1 * 800, 500), resample=Image.BILINEAR))
    plt.title(f"Epoch = {epoch + 1},noisyImages")
    plt.show()
    plt.imshow(show_images(noise_pred).resize((1 * 800, 500), resample=Image.BILINEAR))
    plt.title(f"Epoch = {epoch + 1},predictedNoise,loss={loss}")
    plt.show()
    plt.imshow(show_images(noise).resize((1 * 800, 500), resample=Image.BILINEAR))
    plt.title(f"Epoch = {epoch + 1},pureNoise")
    plt.show()
    # 每五轮输出一次loss
    if (epoch + 1) % 2 == 0:
        loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
        print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")


end_time = time.time()  # stop timer at the end of each 5 epochs
elapsed_time = end_time - start_time

print(elapsed_time) # 195.62777495384216 # 3.26min

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(losses)
axs[1].plot(np.log(losses))
plt.show()

# Step6: Generate Images

# 1. create a pipeline
from diffusers import DDPMPipeline
# 利用训练好的Unet网络，构建用于采样的DDPM模型
image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)

pipeline_output = image_pipe()
plt.imshow(pipeline_output.images[0])
plt.show()
pipeline_output.images[0].size


# 生成随机噪声 随机生成8张噪声图片
sample = torch.randn(8, 3, 32, 32).to(device)

# 
for i, t in enumerate(noise_scheduler.timesteps):
    # 利用模型，根据噪声图像和当前噪声图像所在的步长，预测前一时刻对该时刻添加了多少噪声
    with torch.no_grad():
        residual = model(sample, t).sample

    # 利用去噪公式，对纯噪声图片进行Denoise操作
    sample = noise_scheduler.step(residual, t, sample).prev_sample

plt.imshow(show_images(sample))
plt.show()


# Scaling up with Accelerate

from huggingface_hub import notebook_login

notebook_login() # huggingface-cli login  # login in from the terminal

# !sudo apt -qq install git-lfs
# !git config --global credential.helper store

