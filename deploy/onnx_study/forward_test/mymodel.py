import cv2
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm


class MyModel(nn.Module):
    def __int__(self):
        super(MyModel, self).__int__()
        self.conv = nn.Conv2d(3, 10, 3, 0)

    def forward(self, x):
        x1, x2 = x
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        return x1, x2


dir = Path("../deploy")
mask_list = dir.iterdir()

img = cv2.imread("../deploy/face_ort_3.png", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
h0, w0 = img.shape[:2]
print(h0, w0)
resized_shape = 640
r = resized_shape / max(h0, w0)  # 缩放的比率
if r != 1:  # 总是缩小尺寸，只有在使用增强训练时才会放大
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
h, w = img.shape[:2]
print(h, w)

