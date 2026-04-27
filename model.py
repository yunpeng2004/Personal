from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import LiteWaveletGDU, ResNet34Encoder, WaveletRegionKANSPB
from modules.blocks import ConvGNAct, DepthwiseSeparableConv


class FusionHead(nn.Module):
    def __init__(self, state_channels: int):
        super().__init__()
        self.region_proj = nn.ModuleList([nn.Conv2d(state_channels, 16, kernel_size=1) for _ in range(5)])
        self.boundary_proj = nn.ModuleList([nn.Conv2d(state_channels, 8, kernel_size=1) for _ in range(5)])
        self.confidence_proj = nn.ModuleList([nn.Conv2d(state_channels, 8, kernel_size=1) for _ in range(5)])
        self.fuse = nn.Sequential(
            ConvGNAct(5 * 16 + 5 * 8 + 5 * 8, 64, kernel_size=3),
            DepthwiseSeparableConv(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, stages: List[Dict], output_size: Tuple[int, int]):
        region_feats = []
        boundary_feats = []
        confidence_feats = []
        for idx, stage in enumerate(stages):
            r = self.region_proj[idx](stage["components"]["region"])
            b = self.boundary_proj[idx](stage["components"]["boundary"])
            c = self.confidence_proj[idx](stage["components"]["confidence"])

            r = F.interpolate(r, size=output_size, mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=output_size, mode="bilinear", align_corners=False)
            c = F.interpolate(c, size=output_size, mode="bilinear", align_corners=False)

            region_feats.append(r)
            boundary_feats.append(b)
            confidence_feats.append(c)

        fused = torch.cat(region_feats + boundary_feats + confidence_feats, dim=1)
        return self.fuse(fused)


class WKSPBRegionKANNet(nn.Module):
    def __init__(self, use_pretrained_backbone=True, hidden_channels=128, state_channels=16, kan_groups=4, kan_bases=6):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained=use_pretrained_backbone)
        c1, c2, c3, c4 = 64, 128, 256, 512
        self.spb = WaveletRegionKANSPB(c4, hidden_channels, state_channels, kan_groups, kan_bases)
        self.gdu3 = LiteWaveletGDU(c3, state_channels, hidden_channels, kan_groups, kan_bases)
        self.gdu2 = LiteWaveletGDU(c2, state_channels, hidden_channels, kan_groups, kan_bases)
        self.gdu1 = LiteWaveletGDU(c1, state_channels, hidden_channels, kan_groups, kan_bases)
        self.gdu0 = LiteWaveletGDU(64, state_channels, hidden_channels, kan_groups, kan_bases)
        self.head = FusionHead(state_channels=state_channels)

    def forward(self, x: torch.Tensor):
        stem, e1, e2, e3, e4 = self.encoder(x)

        s4 = self.spb(e4)
        s3 = self.gdu3(s4, e3)
        s2 = self.gdu2(s3, e2)
        s1 = self.gdu1(s2, e1)
        s0 = self.gdu0(s1, stem)

        stages = [s4, s3, s2, s1, s0]
        logits = self.head(stages, output_size=x.shape[-2:])
        aux_region_logits = [
            F.interpolate(stage["maps"]["region_logit"], size=x.shape[-2:], mode="bilinear", align_corners=False)
            for stage in stages
        ]
        aux_boundary_logits = [
            F.interpolate(stage["maps"]["boundary_logit"], size=x.shape[-2:], mode="bilinear", align_corners=False)
            for stage in stages
        ]
        aux_confidence_logits = [
            F.interpolate(stage["maps"]["confidence_logit"], size=x.shape[-2:], mode="bilinear", align_corners=False)
            for stage in stages
        ]

        return {
            "logits": logits,
            "aux_region_logits": aux_region_logits,
            "aux_boundary_logits": aux_boundary_logits,
            "aux_confidence_logits": aux_confidence_logits,
            "stages": stages,
        }
