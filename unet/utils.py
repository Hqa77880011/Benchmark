import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        from medpy import metric
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, float('inf')

from PIL import Image
import os
def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1):
    # 去掉 batch 维
    image = image.squeeze(0).cpu().numpy()  # (H,W,C) or (C,H,W)
    label = label.squeeze(0).cpu().numpy()  # (H,W)


    # 如果是 (H,W,3)，转为 (3,H,W)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = np.transpose(image, (2, 0, 1))  # (3,H,W)

    # 调整大小
    c, x, y = image.shape
    if x != patch_size[0] or y != patch_size[1]:
        img_resized = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)
    else:
        img_resized = image

    # 转 tensor
    device = next(net.parameters()).device
    input_tensor = torch.from_numpy(img_resized).unsqueeze(0).float().to(device)

    net.eval()
    with torch.no_grad():
        outputs = net(input_tensor)
        pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

    # 恢复到原大小
    if pred.shape != label.shape:
        pred = zoom(pred, (label.shape[0] / pred.shape[0], label.shape[1] / pred.shape[1]), order=0)

    # 计算指标
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(pred == i, label == i))

    # 保存彩色预测图
    if test_save_path is not None:
        img_pil = Image.fromarray(pred.astype(np.uint8))
        img_pil.save(os.path.join(test_save_path, f"{case}.png"))

    return metric_list
