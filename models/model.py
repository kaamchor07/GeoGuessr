"""
model.py — Step 2, PRD Section 4 items 1-6

Frozen CLIP ViT-B backbone with:
  - Geocell head: haversine-smoothed cross-entropy (soft targets)
  - Country head: standard cross-entropy classifier
  - Aux heads: Köppen (CE), WorldCover (CE), elevation (MSE regression)
  - Domain-adversarial head: gradient reversal branch (GRL)

The backbone is NEVER updated (requires_grad=False on all backbone params).
Only the head parameters participate in gradient updates.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for domain-adversarial training)
# ---------------------------------------------------------------------------
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (alpha,) = ctx.saved_tensors
        return -alpha * grad_output, None


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer.
    alpha: scaling factor (typically annealed from 0 -> 1 during training).
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha


# ---------------------------------------------------------------------------
# Haversine-smoothed soft target computation
# ---------------------------------------------------------------------------
def build_haversine_soft_targets(
    lats: torch.Tensor,
    lons: torch.Tensor,
    centroid_lats: torch.Tensor,
    centroid_lons: torch.Tensor,
    sigma_km: float = 500.0,
) -> torch.Tensor:
    """
    For each image in the batch, compute a soft probability distribution over
    all geocell centroids using haversine distance decay:
        p(cell_j | img_i) ∝ exp(-d(img_i, centroid_j)² / (2*sigma²))

    Args:
        lats, lons: [B] true locations of batch images
        centroid_lats, centroid_lons: [N] geocell centroid coordinates
        sigma_km: temperature in km (larger = softer targets)

    Returns:
        [B, N] soft target distributions (sum to 1 per row)
    """
    R = 6371.0

    # Expand for broadcasting: [B, 1] and [1, N]
    lat1 = lats.unsqueeze(1).float()   # [B, 1]
    lon1 = lons.unsqueeze(1).float()
    lat2 = centroid_lats.unsqueeze(0).float()  # [1, N]
    lon2 = centroid_lons.unsqueeze(0).float()

    lat1_r = torch.deg2rad(lat1)
    lon1_r = torch.deg2rad(lon1)
    lat2_r = torch.deg2rad(lat2)
    lon2_r = torch.deg2rad(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_r) * torch.cos(lat2_r) * torch.sin(dlon / 2) ** 2
    dist_km = 2 * R * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))  # [B, N]

    weights = torch.exp(-(dist_km ** 2) / (2 * sigma_km ** 2))
    soft_targets = weights / weights.sum(dim=1, keepdim=True)
    return soft_targets


# ---------------------------------------------------------------------------
# Haversine-smoothed cross-entropy loss
# ---------------------------------------------------------------------------
def haversine_soft_ce_loss(
    logits: torch.Tensor,       # [B, N]
    soft_targets: torch.Tensor, # [B, N]
) -> torch.Tensor:
    """
    Cross-entropy loss with soft targets:
        L = -sum_j(soft_targets_j * log_softmax(logits)_j)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    return loss.mean()


# ---------------------------------------------------------------------------
# Head definitions
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Simple 2-layer MLP head."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Stub backbone (fallback when CLIP weights unavailable for dry-run)
# ---------------------------------------------------------------------------
class _StubBackbone(nn.Module):
    """MobileNetV3-small wrapped to output [B, out_dim] like CLIP visual."""
    def __init__(self, mobilenet, out_dim: int):
        super().__init__()
        self.features  = mobilenet.features
        self.avgpool   = nn.AdaptiveAvgPool2d(1)
        in_features    = 576  # MobileNetV3-small features output channels
        self.proj      = nn.Linear(in_features, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class GeoLocModel(nn.Module):
    """
    Frozen CLIP ViT-B backbone + multi-task heads for geolocation.

    Heads:
      - geocell_head:    [embed_dim -> n_geocells]   (haversine-smoothed CE)
      - country_head:    [embed_dim -> n_countries]  (standard CE)
      - koppen_head:     [embed_dim -> 31]            (CE, 30 classes + unknown)
      - worldcover_head: [embed_dim -> 12]            (CE, 11 classes + unknown)
      - elevation_head:  [embed_dim -> 1]             (regression, MSE)
      - domain_head:     [embed_dim -> 2]             (CE via GRL, 2 domains)
    """

    KOPPEN_CLASSES    = 31   # 30 + unknown
    WORLDCOVER_CLASSES = 12  # 11 + unknown

    def __init__(
        self,
        n_geocells: int,
        n_countries: int,
        embed_dim: int = 512,        # CLIP ViT-B/32 output dim
        hidden_dim: int = 512,
        dropout: float = 0.2,
        grl_alpha: float = 1.0,
        clip_model_name: str = "ViT-B/32",   # open_clip name
        clip_pretrained: str = "openai",
    ):
        super().__init__()
        self.n_geocells  = n_geocells
        self.n_countries = n_countries
        self.embed_dim   = embed_dim

        # --- Load frozen CLIP backbone ---
        self._load_clip(clip_model_name, clip_pretrained)

        # --- Projection: optionally add a trainable projector ---
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Task heads ---
        self.geocell_head    = MLP(embed_dim, hidden_dim, n_geocells,           dropout)
        self.country_head    = MLP(embed_dim, hidden_dim, n_countries,          dropout)
        self.koppen_head     = MLP(embed_dim, hidden_dim, self.KOPPEN_CLASSES,  dropout)
        self.worldcover_head = MLP(embed_dim, hidden_dim, self.WORLDCOVER_CLASSES, dropout)
        self.elevation_head  = MLP(embed_dim, hidden_dim // 2, 1,               dropout)

        # --- Domain-adversarial head ---
        self.grl   = GradientReversal(alpha=grl_alpha)
        self.domain_head = MLP(embed_dim, hidden_dim // 2, 2, dropout)

    def _load_clip(self, model_name: str, pretrained: str):
        """Load CLIP via open_clip; freeze all backbone parameters.
        Tries multiple name formats and falls back to a stub for CPU dry runs.
        """
        # Normalise name: open_clip uses 'ViT-B-32', not 'ViT-B/32'
        oc_name = model_name.replace("/", "-")

        loaded = False
        try:
            import open_clip
            # Try with the supplied name first, then normalised
            for name in [oc_name, model_name]:
                try:
                    model_obj, _, _ = open_clip.create_model_and_transforms(
                        name, pretrained=pretrained
                    )
                    self.backbone = model_obj.visual
                    loaded = True
                    print(f"[GeoLocModel] open_clip backbone '{name}' loaded (pretrained={pretrained})")
                    break
                except Exception:
                    continue
        except ImportError:
            pass

        if not loaded:
            # Fallback: lightweight MobileNetV3 stub (CPU-friendly dry run)
            print("[GeoLocModel] WARNING: CLIP unavailable — using MobileNetV3 stub backbone")
            print("  (predictions will be random; this is only for dry-run schema checks)")
            import torchvision.models as tvm
            base = tvm.mobilenet_v3_small(weights=None)
            # Wrap so it outputs [B, embed_dim]
            self.backbone = _StubBackbone(base, out_dim=self.embed_dim)

        # Freeze backbone — CRITICAL per PRD constraint
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        print(f"[GeoLocModel] Backbone frozen. Embed dim: {self.embed_dim}")

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frozen embeddings. Returns [B, embed_dim]."""
        feats = self.backbone(x)
        # Handle different output shapes:
        # - open_clip ViT: [B, embed_dim] (already projected)
        # - ViT raw tokens: [B, seq_len, dim] -> take CLS
        # - StubBackbone:   [B, embed_dim]
        if feats.dim() == 3:
            feats = feats[:, 0]   # CLS token
        if feats.shape[-1] != self.embed_dim:
            # Some CLIP variants return larger dims; project down
            if not hasattr(self, '_dim_proj'):
                self._dim_proj = nn.Linear(feats.shape[-1], self.embed_dim).to(feats.device)
            feats = self._dim_proj(feats)
        if feats.dtype != torch.float32:
            feats = feats.float()
        return feats

    def forward(self, images: torch.Tensor, grl_alpha: float = None) -> dict:
        """
        Forward pass.

        Args:
            images: [B, 3, 224, 224]
            grl_alpha: override GRL alpha (for annealing)

        Returns:
            dict of logits/predictions for each head
        """
        if grl_alpha is not None:
            self.grl.set_alpha(grl_alpha)

        # Frozen backbone
        with torch.no_grad():
            embeds = self.encode_image(images)  # [B, embed_dim]

        # Trainable projection
        proj = self.projector(embeds)  # [B, embed_dim]

        # Task outputs
        geocell_logits    = self.geocell_head(proj)         # [B, n_geocells]
        country_logits    = self.country_head(proj)         # [B, n_countries]
        koppen_logits     = self.koppen_head(proj)          # [B, 31]
        worldcover_logits = self.worldcover_head(proj)      # [B, 12]
        elevation_pred    = self.elevation_head(proj).squeeze(-1)  # [B]

        # Domain-adversarial (gradient reversed)
        domain_logits = self.domain_head(self.grl(proj))   # [B, 2]

        return {
            "geocell_logits":    geocell_logits,
            "country_logits":    country_logits,
            "koppen_logits":     koppen_logits,
            "worldcover_logits": worldcover_logits,
            "elevation_pred":    elevation_pred,
            "domain_logits":     domain_logits,
            "embeddings":        embeds,  # for FAISS indexing
        }

    def trainable_params(self):
        """Return only the non-frozen parameters for the optimizer."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_params(self):
        total   = sum(p.numel() for p in self.parameters())
        frozen  = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = total - frozen
        return total, frozen, trainable


# ---------------------------------------------------------------------------
# Loss function wrapper
# ---------------------------------------------------------------------------
class GeoLoss(nn.Module):
    """
    Combined multi-task loss with configurable weights.

    geocell:    haversine-smoothed cross-entropy  (main signal)
    country:    standard cross-entropy            (coarse)
    koppen:     cross-entropy                     (auxiliary)
    worldcover: cross-entropy                     (auxiliary)
    elevation:  MSE regression                    (auxiliary)
    domain:     cross-entropy via GRL             (adversarial)
    """

    def __init__(
        self,
        centroid_lats: torch.Tensor,
        centroid_lons: torch.Tensor,
        sigma_km: float = 500.0,
        w_geocell: float    = 1.0,
        w_country: float    = 0.5,
        w_koppen: float     = 0.2,
        w_worldcover: float = 0.2,
        w_elevation: float  = 0.1,
        w_domain: float     = 0.3,
        elev_scale: float   = 1000.0,  # normalise elevation in metres
    ):
        super().__init__()
        self.register_buffer("centroid_lats", centroid_lats.float())
        self.register_buffer("centroid_lons", centroid_lons.float())
        self.sigma_km    = sigma_km
        self.w_geocell   = w_geocell
        self.w_country   = w_country
        self.w_koppen    = w_koppen
        self.w_worldcover = w_worldcover
        self.w_elevation = w_elevation
        self.w_domain    = w_domain
        self.elev_scale  = elev_scale

        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse = nn.MSELoss()

    def forward(self, outputs: dict, batch: dict) -> dict:
        device = outputs["geocell_logits"].device
        lats = batch["latitude"].float().to(device)
        lons = batch["longitude"].float().to(device)

        # 1. Geocell (haversine-smoothed CE)
        soft_targets = build_haversine_soft_targets(
            lats, lons, self.centroid_lats, self.centroid_lons, self.sigma_km
        )
        loss_geocell = haversine_soft_ce_loss(outputs["geocell_logits"], soft_targets)

        # 2. Country
        country_idx = batch["country_idx"].long().to(device)
        loss_country = self.ce(outputs["country_logits"], country_idx)

        # 3. Köppen
        koppen = batch["koppen_code"].long().to(device)
        koppen = torch.clamp(koppen, -1, 30)  # cap to valid range
        loss_koppen = self.ce(outputs["koppen_logits"], koppen) if (koppen != -1).any() else torch.tensor(0.0, device=device)

        # 4. WorldCover
        wc = batch["worldcover_code"].long().to(device)
        wc = torch.clamp(wc, -1, 11)
        loss_wc = self.ce(outputs["worldcover_logits"], wc) if (wc != -1).any() else torch.tensor(0.0, device=device)

        # 5. Elevation (MSE, normalised)
        elev = batch["elevation_m"].float().to(device) / self.elev_scale
        valid_elev = ~torch.isnan(elev) & (elev != 0.0)
        if valid_elev.any():
            loss_elev = self.mse(outputs["elevation_pred"][valid_elev], elev[valid_elev])
        else:
            loss_elev = torch.tensor(0.0, device=device)

        # 6. Domain adversarial
        domain = batch["domain_label"].long().to(device)
        valid_domain = domain != -1
        if valid_domain.any():
            loss_domain = self.ce(outputs["domain_logits"][valid_domain], domain[valid_domain])
        else:
            loss_domain = torch.tensor(0.0, device=device)

        # Total
        total = (
            self.w_geocell    * loss_geocell +
            self.w_country    * loss_country +
            self.w_koppen     * loss_koppen +
            self.w_worldcover * loss_wc +
            self.w_elevation  * loss_elev +
            self.w_domain     * loss_domain
        )

        return {
            "loss":          total,
            "loss_geocell":  loss_geocell,
            "loss_country":  loss_country,
            "loss_koppen":   loss_koppen,
            "loss_worldcover": loss_wc,
            "loss_elevation": loss_elev,
            "loss_domain":   loss_domain,
        }
