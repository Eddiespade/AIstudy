import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BackprojectDepth(nn.Module):
    """
    将深度图像转换为3D点云的层
    :param depth:           深度图                (B 1 H W)
    :param inv_K:           相机内参的逆            (B 4 4)
    :return:                3D点云                (B 4 H*W)
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """
    将 3D 点投影到具有内参 K 和变换矩阵 T 的视图中的层
    :param points:          3D点云                (4 4 H*W)
    :param K:               相机内参               (B 4 4)
    :param T:               视图变换矩阵            (B 4 4)
    :return:                3D投影到另一视图        (B H W 2)
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class SSIM(nn.Module):
    """
    计算一对图像之间的 Structural Similarity (SSIM) 损失， 用于度量重建后的图片和原图的结构相似性
    详解参考： https://blog.csdn.net/Kevin_cc98/article/details/79028507
    """

    def __init__(self):
        super(SSIM, self).__init__()
        # 采用 kernel_size = 3， stride = 1 的窗口做平均池化
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_reprojection_loss(pred, target, isL2):
    """
    计算一批预测图像和目标图像之间的重投影损失 L2损失
    """
    L2 = nn.MSELoss(reduction="mean")
    L2_loss = L2(pred, target)

    l1_loss = torch.abs(target - pred).mean(1, True)

    ssim = SSIM()
    ssim_loss = ssim(pred, target).mean(1, True)
    monodepth2_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    if isL2:
        return L2_loss
        # return L2_loss.detach().cpu().numpy()
    else:
        return monodepth2_loss.mean()
        # return reprojection_loss.mean().detach().cpu().numpy()


def compute_loss(right_image, depth, K_L, K_R, cam_T_cam, isL2):
    """
    :param right_image:     右图                  (B C H W)
    :param depth:           深度图                (B 1 H W)
    :param K_L:             左相机内参             (B 4 4)
    :param K_R:             右相机内参             (B 4 4)
    :param cam_T_cam:       左图到右图的变换矩阵     (B 4 4)
    :param isL2:            是否采用L2损失； True：采用l2损失； False：采用monodepth2损失 ssim+l1
    :return:                左右图重构匹配的损失
    """
    bz, c, h, w = right_image.shape

    inv_K_L = np.linalg.pinv(K_L)
    inv_K_L = torch.from_numpy(inv_K_L)

    backproject_depth = BackprojectDepth(bz, h, w)
    cam_points = backproject_depth(depth, inv_K_L)

    project_3d = Project3D(bz, h, w)
    pix_coords = project_3d(cam_points, K_R, cam_T_cam)
    reject_image = F.grid_sample(right_image, pix_coords, padding_mode="border", align_corners=True)
    return compute_reprojection_loss(reject_image, right_image, isL2)


if __name__ == '__main__':
    K_L = np.array([[[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=np.float32)
    K_L = torch.from_numpy(K_L)
    K_R = np.array([[[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=np.float32)
    K_R = torch.from_numpy(K_R)
    cam_T_cam = np.array([[[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                          [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                          [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                          [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=np.float32)
    cam_T_cam = torch.from_numpy(cam_T_cam)

    left_image = torch.rand((4, 3, 8, 8))
    right_image = torch.rand((4, 3, 8, 8))
    depth = torch.rand((4, 1, 8, 8))

    # 调用及参数说明详见 def compute_loss()
    loss = compute_loss(right_image, depth, K_L, K_R, cam_T_cam, False)
    print(loss)
