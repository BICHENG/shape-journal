"""NLF backbone, GPS weight field and localization head."""

import collections
import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

from nlf.pt import ptu
from nlf.pt.models.util import heatmap_to_image, heatmap_to_metric
from nlf_pt_ops import NLFConfig


class InvertedResidual(nn.Module):
    def __init__(
        self, in_channels, out_channels, expand, kernel, stride, norm_layer, drop_probability,
        *, fused, shifted_padding,
    ):
        super().__init__()
        expanded = int(in_channels * expand)
        layers = collections.OrderedDict()
        conv = partial(Conv2dNormActivation, norm_layer=norm_layer)
        if not fused and expanded != in_channels:
            layers["0"] = conv(in_channels, expanded, kernel_size=1, activation_layer=nn.SiLU)

        # The split weights use numeric keys for convolutions and a named padding layer.
        margin = kernel // 2
        shift = int(shifted_padding)
        layers["padding"] = nn.ZeroPad2d((margin - shift, margin + shift) * 2)
        if fused:
            output = out_channels if expanded == in_channels else expanded
            layers["0"] = conv(in_channels, output, kernel_size=kernel, stride=stride, padding=0, activation_layer=nn.SiLU)
        else:
            layers[str(len(layers) - 1)] = conv(
                expanded, expanded, kernel_size=kernel, stride=stride, groups=expanded, padding=0, activation_layer=nn.SiLU,
            )
            layers[str(len(layers) - 1)] = SqueezeExcitation(
                expanded, max(1, in_channels // 4), activation=partial(nn.SiLU, inplace=True),
            )
        if not fused or expanded != in_channels:
            layers[str(len(layers) - 1)] = conv(expanded, out_channels, kernel_size=1, activation_layer=None)

        self.block = nn.Sequential(layers)
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.stochastic_depth = StochasticDepth(drop_probability, "row")

    def forward(self, x):
        result = self.block(x)
        return self.stochastic_depth(result) + x if self.use_res_connect else result


class EfficientNetV2L(nn.Module):
    """EfficientNetV2-L backbone with the padding convention of the NLF weights."""

    def __init__(self, norm_layer=None, stochastic_depth_prob=0.2, centered_stride=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        settings = (
            # expand, kernel, stride, input, output, repeats, block, shifted padding
            (1, 3, 1, 32, 32, 4, "fused", False),
            (4, 3, 2, 32, 64, 7, "fused", False),
            (4, 3, 2, 64, 96, 7, "fused", False),
            (4, 3, 2, 96, 192, 10, "depthwise", False),
            (6, 3, 1, 192, 224, 19, "depthwise", False),
            (6, 3, 2, 224, 384, 25, "depthwise", centered_stride),
            (6, 3, 1, 384, 640, 7, "depthwise", False),
        )
        layers = collections.OrderedDict((
            ("padding", nn.ZeroPad2d(1)),
            ("0", Conv2dNormActivation(3, 32, stride=2, padding=0, norm_layer=norm_layer, activation_layer=nn.SiLU)),
        ))
        total_blocks = sum(setting[5] for setting in settings)
        block_id = 0
        for expand, kernel, stride, in_ch, out_ch, repeats, kind, shift in settings:
            stage = []
            for index in range(repeats):
                stage.append(InvertedResidual(
                    in_ch if index == 0 else out_ch, out_ch, expand, kernel, stride if index == 0 else 1,
                    norm_layer, stochastic_depth_prob * float(block_id) / total_blocks,
                    fused=kind == "fused", shifted_padding=shift and index == 0,
                ))
                block_id += 1
            layers[str(len(layers) - 1)] = nn.Sequential(*stage)
        layers[str(len(layers) - 1)] = Conv2dNormActivation(
            640, 1280, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.SiLU,
        )
        self.features = nn.Sequential(layers)

    def forward(self, x):
        return self.features(x)


class LearnableFourierFeatures(nn.Module):
    """v0.3.2 GPSNet的可学习傅里叶特征"""
    def __init__(self, in_features: int = 3, out_features: int = 512):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features // 2, bias=False)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.linear(inp)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class GPSNet(nn.Module):
    """v0.3.2 GPSNet结构"""
    def __init__(self, pos_enc_dim: int = 512, hidden_dim: int = 2048, output_dim: int = 1024):
        super().__init__()
        self.factor = 1.0 / math.sqrt(pos_enc_dim)
        self.learnable_fourier = LearnableFourierFeatures(3, pos_enc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.register_buffer('center', torch.zeros(3))
        self.register_buffer('mini', torch.zeros(3))
        self.register_buffer('maxi', torch.ones(3))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = (inp - self.center) / (self.maxi - self.mini)
        x = self.learnable_fourier(x) * self.factor
        return self.mlp(x)


class GPSField(nn.Module):
    """v0.3.2权重场，匹配GPSNet + pred_mlp结构"""
    def __init__(self, posenc_dim: int = 1024, hidden_dim: int = 384, output_dim: int = 5130):
        super().__init__()
        self.posenc_dim = posenc_dim
        self.gps_net = GPSNet(pos_enc_dim=512, hidden_dim=2048, output_dim=posenc_dim)
        self.pred_mlp = nn.Sequential(
            nn.Linear(posenc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.register_buffer('r_sqrt_eigva', torch.ones(posenc_dim))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        batch_shape = inp.shape[:-1]
        flat_inp = inp.reshape(-1, 3)
        lbo = self.gps_net(flat_inp)[..., :self.posenc_dim]
        lbo = lbo.reshape(*batch_shape, self.posenc_dim)
        lbo = lbo * self.r_sqrt_eigva[:self.posenc_dim] * 0.1
        return self.pred_mlp(lbo)


class LocalizerHead(nn.Module):
    """纯PyTorch LocalizerHead"""

    def __init__(self, weight_field: nn.Module, config: NLFConfig):
        super().__init__()
        self.config = config
        self.weight_field = weight_field
        self.layer = nn.Sequential(
            nn.Conv2d(1280, config.backbone_link_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.backbone_link_dim, eps=1e-3, momentum=0.01),
            nn.SiLU(),
        )

    def get_weights_for_canonical_points(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.weight_field(points).half().unflatten(
            -1, (self.config.backbone_link_dim + 1, 2 + self.config.depth),
        )
        return weights[..., :-1, :].permute(0, 2, 1).contiguous(), weights[..., -1, :].contiguous()

    def apply_weights(
        self,
        features: torch.Tensor,
        w_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_out_channels = 2 + self.config.depth

        points, channels, features_in = w_tensor.shape
        w_flat = w_tensor.reshape(points * channels, features_in, 1, 1).contiguous().float()
        b_flat = b_tensor.reshape(-1).float()

        logits = F.conv2d(features, w_flat, bias=b_flat).float()
        logits = torch.unflatten(logits, 1, (-1, n_out_channels))

        uncertainty_map = logits[:, :, 0]
        coords_metric_xy = ptu.soft_argmax(logits[:, :, 1], dim=[3, 2])
        heatmap25d = ptu.softmax(logits[:, :, 2:], dim=[4, 3, 2])
        heatmap2d = torch.sum(heatmap25d, dim=2)

        uncertainties = torch.einsum('nphw,nphw->np', uncertainty_map, heatmap2d.detach())
        uncertainties = F.softplus(uncertainties + self.config.uncert_bias) + self.config.uncert_bias2

        coords25d = ptu.decode_heatmap(heatmap25d, dim=[4, 3, 2])
        coords2d_raw = coords25d[..., :2]
        coords3d_raw = torch.cat([coords_metric_xy, coords25d[..., 2:]], dim=-1)

        coords2d = heatmap_to_image(
            coords2d_raw, self.config.proc_side, self.config.stride_test, self.config.centered_stride
        )
        coords3d = heatmap_to_metric(
            coords3d_raw, self.config.proc_side, self.config.stride_test,
            self.config.centered_stride, self.config.box_size_m
        )

        return coords2d, coords3d, uncertainties
