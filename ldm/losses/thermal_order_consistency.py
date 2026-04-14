"""
Thermal Order Consistency Loss (L_TOC) for PID model.

Borrowed from Real-IISR (arXiv:2603.04745) and adapted for the PID
Physics-Informed Diffusion framework.

Core idea:
  In infrared imaging, higher temperature → higher pixel intensity.
  This is a monotonic relationship. L_TOC does NOT care about absolute
  brightness values; it only penalizes cases where the *relative brightness
  order* between neighbouring patches is reversed between the predicted
  image and the ground-truth.

  L_TOC = (1 / |Ω|) Σ_{(i,j)∈Ω} ReLU( −(SR_i − SR_j) × (HR_i − HR_j) )

  where Ω is the set of spatially adjacent patch pairs.

Integration point:
  This file is imported in `ldm/models/diffusion/ddpm_tev.py`.
  The loss is computed on the decoded pixel-space images (x̂₀ and x₀)
  and added as a weighted term alongside L_Rec and L_TeV.

Usage:
  from thermal_order_consistency import ThermalOrderConsistencyLoss
  toc_loss_fn = ThermalOrderConsistencyLoss(patch_size=8)
  loss_toc = toc_loss_fn(pred_ir, gt_ir)   # both [B, C, H, W], range [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalOrderConsistencyLoss(nn.Module):
    """
    Thermal Order Consistency Loss.

    Enforces monotonic brightness ordering between adjacent patches
    in the predicted infrared image, guided by the ground truth.

    Parameters
    ----------
    patch_size : int
        Side length of each patch (pixels). Patches are obtained via
        average pooling with this kernel size. Default 8, following
        Real-IISR.  Larger patches are more tolerant of noise but
        less sensitive to fine-grained ordering; smaller patches give
        stronger per-pixel supervision but are noisier.
    """

    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : [B, C, H, W]  predicted infrared image, range [0, 1]
        target : [B, C, H, W]  ground-truth infrared image, range [0, 1]

        Returns
        -------
        loss : scalar tensor
        """
        # ---- 1. Convert to single-channel grayscale ----
        # PID's IR images are stored as 3-channel (RGB duplicated).
        # Taking the mean collapses them to a single thermal intensity.
        pred_gray = pred.mean(dim=1, keepdim=True)      # [B, 1, H, W]
        target_gray = target.mean(dim=1, keepdim=True)   # [B, 1, H, W]

        # ---- 2. Compute patch-level mean intensities ----
        pred_patches = F.avg_pool2d(pred_gray, self.patch_size)    # [B, 1, H', W']
        target_patches = F.avg_pool2d(target_gray, self.patch_size)

        # ---- 3. Compute pairwise differences among adjacent patches ----
        # Horizontal neighbours (left-right)
        pred_diff_h = pred_patches[:, :, :, 1:] - pred_patches[:, :, :, :-1]
        target_diff_h = target_patches[:, :, :, 1:] - target_patches[:, :, :, :-1]

        # Vertical neighbours (top-bottom)
        pred_diff_v = pred_patches[:, :, 1:, :] - pred_patches[:, :, :-1, :]
        target_diff_v = target_patches[:, :, 1:, :] - target_patches[:, :, :-1, :]

        # ---- 4. Penalise order reversals ----
        # If pred says A > B but target says A < B, the product is negative
        # → ReLU(-product) > 0 → penalty.
        # If orderings agree, product ≥ 0 → ReLU(-product) = 0 → no penalty.
        loss_h = F.relu(-pred_diff_h * target_diff_h)
        loss_v = F.relu(-pred_diff_v * target_diff_v)

        # ---- 5. Average over all patch pairs ----
        loss = (loss_h.mean() + loss_v.mean()) / 2.0

        return loss


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 256, 256

    # Case 1: identical images → loss should be 0
    img = torch.rand(B, C, H, W)
    fn = ThermalOrderConsistencyLoss(patch_size=8)
    loss_zero = fn(img, img)
    print(f"[Test] Identical images  → L_TOC = {loss_zero.item():.6f}  (expect ~0)")

    # Case 2: inverted image → loss should be large
    img_inv = 1.0 - img
    loss_inv = fn(img_inv, img)
    print(f"[Test] Inverted images   → L_TOC = {loss_inv.item():.6f}  (expect >> 0)")

    # Case 3: slight perturbation → loss should be very small
    img_noisy = img + 0.01 * torch.randn_like(img)
    img_noisy.clamp_(0, 1)
    loss_noisy = fn(img_noisy, img)
    print(f"[Test] Slightly noisy    → L_TOC = {loss_noisy.item():.6f}  (expect ~0)")

    print("\nAll sanity checks passed.")
