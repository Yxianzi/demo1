"""SR-SSFTT model modules.

This file implements a compact SSFTT-style backbone adapted to the Houston
patch format used by this project. The design follows the public SSFTT idea
(3D CNN + 2D CNN + spectral-spatial tokenization + Transformer) but rewrites
the modules locally and adds three project-specific components:

1. SpatialReliabilityTokenizer
2. DomainAdaptiveAdapter inside the Transformer stack
3. PrototypeGuidedHead
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class SpectralSpatialStem(nn.Module):
    """3D spectral-spatial stem followed by 2D spatial refinement.

    Args:
        n_band: Number of hyperspectral bands.
        out_dim: Output feature-map channel dimension.

    Input shape:
        x: [B, n_band, H, W]

    Output shape:
        feature_map: [B, out_dim, H, W]
    """

    def __init__(self, n_band, out_dim=64):
        super().__init__()
        self.n_band = n_band
        self.spectral3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.spatial2d = nn.Sequential(
            nn.Conv2d(16 * n_band, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b = x.size(0)
        x = x.unsqueeze(1)
        x = self.spectral3d(x)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(b, 16 * self.n_band, x.size(2), x.size(3))
        return self.spatial2d(x)


class SpatialReliabilityTokenizer(nn.Module):
    """Reliability-guided tokenization over a spectral-spatial feature map.

    If reliability and reliability_map are both None, this is a normal
    learnable tokenizer. A sample-level reliability vector [B] is kept for
    compatibility and acts as sample-level token gating. A spatial reliability
    map [B, H*W] or [B, H, W] directly adjusts token logits before softmax and
    therefore changes spatial attention distribution.

    Input:
        feature_map: [B, D, H, W]
        reliability: [B] or None
        reliability_map: [B, H*W], [B, H, W], or None

    Output:
        tokens: [B, num_tokens, token_dim]
    """

    def __init__(self, in_dim, num_tokens=4, token_dim=64, use_reliability=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.use_reliability = use_reliability
        self.token_queries = nn.Parameter(torch.randn(num_tokens, in_dim) * 0.02)
        self.token_proj = nn.Linear(in_dim, token_dim)
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, feature_map, reliability=None, reliability_map=None):
        b, d, h, w = feature_map.shape
        patches = feature_map.flatten(2).transpose(1, 2)
        logits = torch.matmul(patches, self.token_queries.t()) / math.sqrt(float(d))
        token_logits = logits.transpose(1, 2)

        if self.use_reliability and reliability_map is not None:
            gamma = F.softplus(self.gamma)
            rel_map = reliability_map.to(feature_map.device, feature_map.dtype)
            if rel_map.dim() == 3:
                rel_map = rel_map.flatten(1)
            if rel_map.dim() != 2 or rel_map.size(0) != b or rel_map.size(1) != h * w:
                raise ValueError("reliability_map must have shape [B, H*W] or [B, H, W].")
            token_logits = token_logits + gamma * rel_map.clamp_min(0.0).unsqueeze(1)

        token_weight = torch.softmax(token_logits, dim=-1)

        if self.use_reliability and reliability is not None:
            gamma = F.softplus(self.gamma)
            # Sample-level reliability is a coarse compatibility path. It gates
            # all spatial locations equally and does not reshape attention like
            # reliability_map does.
            rel = reliability.to(feature_map.device, feature_map.dtype).view(b, 1, 1)
            token_weight = token_weight * (1.0 + gamma * rel.clamp_min(0.0))

        tokens = torch.matmul(token_weight, patches)
        return self.token_proj(tokens)


class DomainAdaptiveAdapter(nn.Module):
    """Bottleneck token adapter with separate source/target residual scales."""

    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.source_scale = nn.Parameter(torch.tensor(1.0))
        self.target_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, tokens, domain="source"):
        residual = self.up(F.relu(self.down(tokens), inplace=True))
        scale = self.target_scale if domain == "target" else self.source_scale
        return tokens + scale * residual


class TransformerEncoderWithDomainAdapter(nn.Module):
    """Transformer encoder with a domain-adaptive adapter after each block."""

    def __init__(
        self,
        dim=64,
        depth=2,
        num_heads=4,
        adapter_bottleneck=64,
        use_domain_adapter=True,
    ):
        super().__init__()
        self.use_domain_adapter = use_domain_adapter
        self.layers = nn.ModuleList()
        self.adapters = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
            )
            self.adapters.append(DomainAdaptiveAdapter(dim, adapter_bottleneck))
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, domain="source"):
        for layer, adapter in zip(self.layers, self.adapters):
            tokens = layer(tokens)
            if self.use_domain_adapter:
                tokens = adapter(tokens, domain=domain)
        return self.norm(tokens)


class PrototypeGuidedHead(nn.Module):
    """Linear classifier plus optional cosine prototype logits."""

    def __init__(
        self,
        feature_dim,
        num_classes,
        temperature=0.1,
        proto_logit_weight=0.5,
        use_proto_head=True,
    ):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.temperature = temperature
        self.use_proto_head = use_proto_head
        self.proto_logit_weight = nn.Parameter(torch.tensor(float(proto_logit_weight)))

    def forward(self, features, prototypes=None):
        linear_logits = self.classifier(features)
        proto_logits = features.new_zeros(linear_logits.shape)

        if self.use_proto_head and prototypes is not None:
            prototypes = prototypes.to(device=features.device, dtype=features.dtype)
            norm_features = F.normalize(features, p=2, dim=1, eps=1e-8)
            norm_prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-8)
            proto_logits = torch.matmul(norm_features, norm_prototypes.t())
            proto_logits = proto_logits / max(float(self.temperature), 1e-6)
            logits = linear_logits + self.proto_logit_weight * proto_logits
        else:
            logits = linear_logits

        return logits, linear_logits, proto_logits


class SRSSFTT(nn.Module):
    """Spatial-Reliability Guided SSFTT.

    Args:
        n_band: Number of input spectral bands.
        num_classes: Number of land-cover classes.
        patch_size: Input patch width/height. Kept for config compatibility.

    Forward input:
        x: [B, n_band, patch_size, patch_size]
        reliability: [B] or None
        domain: "source" or "target"
        prototypes: [num_classes, feature_dim] or None

    Returns:
        dict with features, tokens, logits, linear_logits, proto_logits.
    """

    def __init__(
        self,
        n_band,
        num_classes,
        patch_size=7,
        num_tokens=4,
        token_dim=64,
        transformer_depth=2,
        num_heads=4,
        adapter_bottleneck=64,
        proto_temperature=0.1,
        proto_logit_weight=0.5,
        use_reliability_tokenizer=True,
        use_domain_adapter=True,
        use_proto_head=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = token_dim
        self.stem = SpectralSpatialStem(n_band=n_band, out_dim=token_dim)
        self.tokenizer = SpatialReliabilityTokenizer(
            in_dim=token_dim,
            num_tokens=num_tokens,
            token_dim=token_dim,
            use_reliability=use_reliability_tokenizer,
        )
        self.encoder = TransformerEncoderWithDomainAdapter(
            dim=token_dim,
            depth=transformer_depth,
            num_heads=num_heads,
            adapter_bottleneck=adapter_bottleneck,
            use_domain_adapter=use_domain_adapter,
        )
        self.head = PrototypeGuidedHead(
            feature_dim=token_dim,
            num_classes=num_classes,
            temperature=proto_temperature,
            proto_logit_weight=proto_logit_weight,
            use_proto_head=use_proto_head,
        )

    def forward(self, x, reliability=None, reliability_map=None, domain="source", prototypes=None):
        feature_map = self.stem(x)
        tokens = self.tokenizer(
            feature_map,
            reliability=reliability,
            reliability_map=reliability_map,
        )
        tokens = self.encoder(tokens, domain=domain)
        features = tokens.mean(dim=1)
        logits, linear_logits, proto_logits = self.head(features, prototypes=prototypes)
        return {
            "features": features,
            "tokens": tokens,
            "logits": logits,
            "linear_logits": linear_logits,
            "proto_logits": proto_logits,
        }
