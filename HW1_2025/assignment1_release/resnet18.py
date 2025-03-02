'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        # Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(planes)
           )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Uncomment the following lines and replace the ? with correct values
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        out = self.conv1(images)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.flatten(start_dim=1)
        out = self.linear(out)
        return out

    def visualize(self, logdir):
        import os, matplotlib.pyplot as plt, numpy as np
        weights = self.conv1.weight.detach().cpu().numpy()

        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < weights.shape[0]:
                kernel = weights[i].transpose(1, 2, 0)
                min_val = kernel.min()
                max_val = kernel.max()
                diff = max_val - min_val
                if diff > 0:
                    kernel = (kernel - min_val) / diff
                ax.imshow(kernel)
            ax.set_xticks([])
            ax.set_yticks([])
            row, col = i // 8, i % 8
            if row == 7:
                ax.text(0.5, -0.15, str(col + 1), transform=ax.transAxes, ha='center', va='top', fontsize=8)
            if col == 0:
                ax.text(-0.15, 0.5, str(row + 1), transform=ax.transAxes, ha='right', va='center', fontsize=8)
        plt.tight_layout()
        save_path = os.path.join(logdir, "kernels.png")
        plt.savefig(save_path)
        plt.close(fig)

        fig_gray, axes_gray = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes_gray.flat):
            if i < weights.shape[0]:
                kernel = weights[i].transpose(1, 2, 0)
                kernel_gray = np.dot(kernel, [0.2989, 0.5870, 0.1140])
                min_val = kernel_gray.min()
                max_val = kernel_gray.max()
                diff = max_val - min_val
                if diff > 0:
                    kernel_gray = (kernel_gray - min_val) / diff
                ax.imshow(kernel_gray, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            row, col = i // 8, i % 8
            if row == 7:
                ax.text(0.5, -0.15, str(col + 1), transform=ax.transAxes, ha='center', va='top', fontsize=8)
            if col == 0:
                ax.text(-0.15, 0.5, str(row + 1), transform=ax.transAxes, ha='right', va='center', fontsize=8)
        plt.tight_layout()
        gray_path = os.path.join(logdir, "kernels_gray.png")
        plt.savefig(gray_path)
        plt.close(fig_gray)

        fig_avg, axes_avg = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes_avg.flat):
            if i < weights.shape[0]:
                kernel = weights[i].transpose(1, 2, 0)
                kernel_avg = kernel.mean(axis=-1)
                min_val = kernel_avg.min()
                max_val = kernel_avg.max()
                diff = max_val - min_val
                if diff > 0:
                    kernel_avg = (kernel_avg - min_val) / diff
                ax.imshow(kernel_avg, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            row, col = i // 8, i % 8
            if row == 7:
                ax.text(0.5, -0.15, str(col + 1), transform=ax.transAxes, ha='center', va='top', fontsize=8)
            if col == 0:
                ax.text(-0.15, 0.5, str(row + 1), transform=ax.transAxes, ha='right', va='center', fontsize=8)
        plt.tight_layout()
        avg_path = os.path.join(logdir, "kernels_avg.png")
        plt.savefig(avg_path)
        plt.close(fig_avg)

        fig_chan, axes_chan = plt.subplots(8, 3, figsize=(6, 16))
        for i in range(8):
            for j in range(3):
                ax = axes_chan[i, j]
                channel_img = weights[i][j]
                min_val = channel_img.min()
                max_val = channel_img.max()
                diff = max_val - min_val
                if diff > 0:
                    channel_img = (channel_img - min_val) / diff
                ax.imshow(channel_img, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    if j == 0:
                        ax.set_title("Red", fontsize=16)
                    elif j == 1:
                        ax.set_title("Green", fontsize=16)
                    elif j == 2:
                        ax.set_title("Blue", fontsize=16)
            axes_chan[i, 0].set_ylabel(f"(1,{i + 1})", fontsize=16)
        plt.tight_layout()
        chan_path = os.path.join(logdir, "kernels_channels.png")
        plt.savefig(chan_path)
        plt.close(fig_chan)
