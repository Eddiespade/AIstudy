import os
import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn


# 定义的超分辨率网络结构
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)
    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint 
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


if __name__ == '__main__':

    # 下载权重文件.pth 和 测试图片

    urls = [
        'https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
        'https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg.mp.itc.cn%2Fupload%2F20170604%2F10c6fad2080f438bae1ce98b0d07eed2_th.jpg&refer=http%3A%2F%2Fimg.mp.itc.cn&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1660440355&t=ba210d286709983734a43f3451c4629e']
    names = ['srcnn.pth', 'face.png']
    for url, name in zip(urls, names):
        if not os.path.exists(name):
            open(name, 'wb').write(requests.get(url).content)

    model = init_torch_model()
    input_img = cv2.imread('face.png').astype(np.float32)

    # HWC to NCHW
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = np.expand_dims(input_img, 0)

    # Inference
    torch_output = model(torch.from_numpy(input_img)).detach().numpy()

    # NCHW to HWC
    torch_output = np.squeeze(torch_output, 0)
    torch_output = np.clip(torch_output, 0, 255)
    torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

    # Show image
    cv2.imwrite("face_torch.png", torch_output)

    # torch2onnx
    x = torch.randn(1, 3, 718, 640)

    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            "srcnn.onnx",
            opset_version=11,
            input_names=['input'],
            output_names=['output'])
