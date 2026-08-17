"""
noise_aug.py — Step 3, PRD Section 3 & Section 4

Implements matched noise augmentation based on the findings from Step 1 data audit:
  - StreetView images have higher noise (HF residual std ~18.8 vs ~15.0 on dashcam)
  - Simulates high-frequency sensor grain / compression artifacts so external data
    matches the noise distribution of the evaluation set.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


class MatchedNoiseAugmentation:
    """
    Applies synthetic noise and compression artifacts to match the noised evaluation set.
    """

    def __init__(self, target_noise_std: float = 18.0, p_apply: float = 0.5):
        self.target_noise_std = target_noise_std
        self.p_apply = p_apply

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p_apply:
            return img

        arr = np.array(img, dtype=np.float32)
        
        # 1. Add Gaussian grain calibrated to audit findings
        sigma = np.random.uniform(5.0, self.target_noise_std)
        noise = np.random.normal(0, sigma, arr.shape)
        noised = np.clip(arr + noise, 0, 255).astype(np.uint8)
        
        res_img = Image.fromarray(noised)

        # 2. Random mild JPEG compression artifact (quality 65-90)
        if np.random.rand() < 0.4:
            import io
            buf = io.BytesIO()
            q = int(np.random.randint(65, 90))
            res_img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            res_img = Image.open(buf).convert("RGB")

        return res_img
