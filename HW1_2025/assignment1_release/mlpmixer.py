import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # Uncomment this line and replace ? with correct values
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch. num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0),
            activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # norm1 used with mlp_tokens
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # norm2 used with mlp_channels
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        out = self.norm1(x)
        out = self.mlp_tokens(out.permute(0, 2, 1)).permute(0, 2, 1) + x
        out2 = self.norm2(out)
        out2 = self.mlp_channels(out2) + out
        return out2


class MLPMixer(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, 
                 drop_rate=0., activation='gelu'):
        super(MLPMixer, self).__init__()
        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=3,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                activation=activation, drop=drop_rate)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """ MLPMixer forward
        :param images: [batch, 3, img_size, img_size]
        """

        out = self.patchemb(images)
        out = self.blocks(out)
        out = self.norm(out)
        out = out.mean(dim=1)
        out = self.head(out)
        return out

    def visualize(self, logdir):
        """
        Visualize paired kernels of the token-mixing MLP weights from the first Mixer block.

        For the first block, the fc1 weights (shape: [num_units, num_weights]) are reshaped into
        sqrt(num_weights) x sqrt(num_weights) kernels (e.g., 64 weights -> 8x8 kernel), normalized,
        sorted by a frequency metric (sum of absolute values as a proxy), and paired (first with last,
        second with second-last, etc.) to highlight opposing phases. A montage is then created and saved.
        """

        token_mixing_layer = self.blocks[0].mlp_tokens.fc1
        weights = token_mixing_layer.weight.detach().cpu().numpy()

        kernels = []
        for w in weights:
            num_weights = w.size
            side = int(np.sqrt(num_weights))
            kernel = w.reshape(side, side)
            # normalize
            kernel = kernel - kernel.mean()
            max_abs = np.abs(kernel).max()
            if max_abs > 0:
                kernel = kernel / max_abs
            kernels.append(kernel)

        def compute_freq_metric(kernel):
            # compute 2D Fourier transform and shift the zero frequency component to the center
            f = np.fft.fft2(kernel)
            f = np.fft.fftshift(f)
            # radial grid
            rows, cols = kernel.shape
            y = np.arange(rows) - rows // 2
            x = np.arange(cols) - cols // 2
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X ** 2 + Y ** 2)
            # final metric: sum of (radial distance * magnitude of Fourier coefficient)
            return np.sum(R * np.abs(f))

        # compute frequency metric for each kernel
        freq_metrics = [compute_freq_metric(kernel) for kernel in kernels]

        # sort kernels per frequency
        sorted_indices = np.argsort(freq_metrics)
        sorted_kernels = [kernels[i] for i in sorted_indices]

        # plotting
        num_kernels = len(sorted_kernels)
        cols = int(np.ceil(np.sqrt(num_kernels)))
        rows = int(np.ceil(num_kernels / cols))

        kernel_size = sorted_kernels[0].shape[0]
        vertical_gap = 1
        horizontal_gap = 1

        figure_height = rows * kernel_size + (rows - 1) * vertical_gap
        figure_width = cols * kernel_size + (cols - 1) * horizontal_gap

        figure = np.zeros((figure_height, figure_width))

        for idx, kernel in enumerate(sorted_kernels):
            r = idx // cols
            c = idx % cols
            y = r * (kernel_size + vertical_gap)
            x = c * (kernel_size + horizontal_gap)
            figure[y:y + kernel_size, x:x + kernel_size] = kernel

        # padding
        padded_figure = np.zeros((figure_height + 2, figure_width + 2)) * (-1)
        padded_figure[1:-1, 1:-1] = figure

        # custom colors
        custom_cmap = LinearSegmentedColormap.from_list(
            'custom_bwr_dark',
            [(0.0, (0, 0, 0.65)), (0.5, (1, 1, 1)), (1.0, (0.65, 0, 0))]
        )

        fig_path = os.path.join(logdir, "token_mixing_block1_per_freq.png")
        plt.figure(figsize=(cols * 2, rows * 2))
        plt.imshow(padded_figure, cmap=custom_cmap, interpolation='nearest')

        ax = plt.gca()
        for r in range(rows):
            center_y = 1 + r * (kernel_size + vertical_gap) + kernel_size / 2
            ax.text(-kernel_size / 2, center_y, str(r + 1), va='center', ha='center',
                    fontsize=24, color='black')
        for c in range(cols):
            center_x = 1 + c * (kernel_size + horizontal_gap) + kernel_size / 2
            ax.text(center_x, -kernel_size / 2, str(c + 1), va='center', ha='center',
                    fontsize=24, color='black')

        plt.xlim(-kernel_size, figure_width + 2)
        plt.ylim(figure_height + 2, -kernel_size)
        plt.axis('off')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        print(f'Visualization saved to {fig_path}')
 
