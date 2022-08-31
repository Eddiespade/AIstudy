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

        # np.meshgrid()：从坐标向量中返回坐标矩阵; indexing：输出的笛卡尔（默认为“ xy”）或矩阵（“ ij”）索引。
        # 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # 作为nn.Module中的可训练参数使用，但参数不更新
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)
        # 对应尺寸的全 1 参数
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)

        # 将x, y 所有取值在 0 维拼接，然后再在拼接后的 0维增加一个长度为1的维度
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


def compute_reprojection_loss(pred, target):
    """
    计算一批预测图像和目标图像之间的重投影损失 L2损失
    """
    L2_loss = nn.MSELoss(reduction="mean")
    loss = L2_loss(pred, target)
    # return loss.detach().cpu().numpy()
    return loss

def compute_loss(right_image, depth, K_L, K_R, cam_T_cam):
    """
    :param left_image:      左图                  (B 1 H W)
    :param right_image:     右图                  (B 1 H W)
    :param depth:           深度图                (B 1 H W)
    :param K_L:             左相机内参             (B 4 4)
    :param K_R:             右相机内参             (B 4 4)
    :param cam_T_cam:       左图到右图的变换矩阵     (B 4 4)
    :return:                左右图重构匹配的损失
    """
    # 获取B，C，H，W
    bz, c, h, w = right_image.shape

    # 1. 求相机内参的逆矩阵
    inv_K_L = np.linalg.pinv(K_L)
    inv_K_L = torch.from_numpy(inv_K_L)

    # 2. 将深度图投影成3D点云
    backproject_depth = BackprojectDepth(bz, h, w)
    cam_points = backproject_depth(depth, inv_K_L)

    # 3. 将左视图的3D点云投影成右视图的2D图像
    project_3d = Project3D(bz, h, w)
    pix_coords = project_3d(cam_points, K_R, cam_T_cam)
    reject_image = F.grid_sample(right_image, pix_coords, padding_mode="border", align_corners=True)
    return compute_reprojection_loss(reject_image, right_image)


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

    left_image = torch.rand((4, 1, 8, 8))
    right_image = torch.rand((4, 1, 8, 8))
    depth = torch.rand((4, 1, 8, 8))

    loss = compute_loss(right_image, depth, K_L, K_R, cam_T_cam)
    print(loss)
