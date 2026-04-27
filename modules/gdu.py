import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DepthwiseSeparableConv
from .sabi_kan import RegionSABIKAN
from .wavelet import dwt2d


class LiteWaveletGDU(nn.Module):
    def __init__(self, enc_channels: int, state_channels: int, hidden_channels: int, kan_groups: int, kan_bases: int):
        super().__init__()
        self.state_channels = state_channels
        self.enc_proj = ConvGNAct(enc_channels, hidden_channels, kernel_size=1)
        self.ll_refine = DepthwiseSeparableConv(hidden_channels, hidden_channels)
        self.hf_fuse = ConvGNAct(hidden_channels * 3, hidden_channels, kernel_size=1)
        self.hf_refine = DepthwiseSeparableConv(hidden_channels, hidden_channels)

        self.sgca_ll = nn.Sequential(
            ConvGNAct(hidden_channels + 2 * state_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sgca_hf = nn.Sequential(
            ConvGNAct(hidden_channels + 2 * state_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.region_kan = RegionSABIKAN(hidden_channels + 2 * state_channels, state_channels, groups=kan_groups, num_bases=kan_bases)
        self.boundary_head = nn.Sequential(
            ConvGNAct(hidden_channels + 2 * state_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1),
        )
        self.confidence_head = nn.Sequential(
            ConvGNAct(2 * hidden_channels + 2 * state_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1),
        )

        self.region_logit = nn.Conv2d(state_channels, 1, kernel_size=1)
        self.boundary_logit = nn.Conv2d(state_channels, 1, kernel_size=1)
        self.confidence_logit = nn.Conv2d(state_channels, 1, kernel_size=1)

    def forward(self, prev_state: dict, enc_feat: torch.Tensor):
        h0, w0 = enc_feat.shape[-2:]
        z = self.enc_proj(enc_feat)

        ll, lh, hl, hh = dwt2d(z)
        ll = self.ll_refine(ll)
        hf = self.hf_refine(self.hf_fuse(torch.cat([lh, hl, hh], dim=1)))

        half_size = ll.shape[-2:]
        prev_region = F.interpolate(prev_state["components"]["region"], size=half_size, mode="bilinear", align_corners=False)
        prev_boundary = F.interpolate(prev_state["components"]["boundary"], size=half_size, mode="bilinear", align_corners=False)
        prev_conf = F.interpolate(prev_state["components"]["confidence"], size=half_size, mode="bilinear", align_corners=False)

        ll_gate = self.sgca_ll(torch.cat([ll, prev_region, prev_conf], dim=1))
        hf_gate = self.sgca_hf(torch.cat([hf, prev_boundary, prev_region], dim=1))
        ll = ll * ll_gate
        hf = hf * hf_gate

        region_half = self.region_kan(torch.cat([ll, prev_region, prev_conf], dim=1), state_hint=prev_region)
        boundary_half = self.boundary_head(torch.cat([hf, prev_boundary, prev_region], dim=1))
        confidence_half = self.confidence_head(torch.cat([ll, hf, prev_conf, prev_region], dim=1))

        region = F.interpolate(region_half, size=(h0, w0), mode="bilinear", align_corners=False)
        boundary = F.interpolate(boundary_half, size=(h0, w0), mode="bilinear", align_corners=False)
        confidence = F.interpolate(confidence_half, size=(h0, w0), mode="bilinear", align_corners=False)

        region_logit = self.region_logit(region)
        boundary_logit = self.boundary_logit(boundary)
        confidence_logit = self.confidence_logit(confidence)

        boundary_prob = torch.sigmoid(boundary_logit)
        confidence_prob = torch.sigmoid(confidence_logit)

        confidence_gate = 0.5 + 0.5 * confidence_prob
        boundary_gate = 1.0 - 0.5 * boundary_prob
        region = region * confidence_gate * boundary_gate
        boundary = boundary * (0.5 + 0.5 * boundary_prob)
        confidence = confidence * confidence_gate

        tensor = torch.cat([region, boundary, confidence], dim=1)
        return {
            "tensor": tensor,
            "components": {
                "region": region,
                "boundary": boundary,
                "confidence": confidence,
            },
            "maps": {
                "region_logit": region_logit,
                "boundary_logit": boundary_logit,
                "confidence_logit": confidence_logit,
            },
        }
