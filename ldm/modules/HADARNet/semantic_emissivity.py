import torch
import torch.nn as nn
import torch.nn.functional as F

# 语义类别到发射率先验的映射表
# 格式: {类别索引: (均值, 标准差)}
# 基于 KAIST/FLIR 场景的 Cityscapes 类别子集
EMISSIVITY_PRIOR = {
    0:  (0.93, 0.03),   # road
    1:  (0.90, 0.05),   # sidewalk
    2:  (0.88, 0.06),   # building
    3:  (0.85, 0.08),   # wall
    4:  (0.87, 0.07),   # fence
    5:  (0.85, 0.05),   # pole
    6:  (0.92, 0.04),   # traffic light
    7:  (0.91, 0.04),   # traffic sign
    8:  (0.96, 0.02),   # vegetation
    9:  (0.95, 0.03),   # terrain
    10: (0.85, 0.10),   # sky
    11: (0.98, 0.01),   # person  ← 关键：人体发射率极高
    12: (0.97, 0.01),   # rider
    13: (0.25, 0.10),   # car     ← 关键：金属车身极低
    14: (0.30, 0.12),   # truck
    15: (0.28, 0.11),   # bus
    16: (0.27, 0.10),   # train
    17: (0.25, 0.10),   # motorcycle
    18: (0.28, 0.09),   # bicycle
}

NUM_CLASSES = len(EMISSIVITY_PRIOR)

# 构建先验张量，shape: [num_classes, 2] (mean, std)
def build_prior_tensors(device):
    means = torch.zeros(NUM_CLASSES, device=device)
    stds  = torch.zeros(NUM_CLASSES, device=device)
    for cls_id, (mu, sigma) in EMISSIVITY_PRIOR.items():
        means[cls_id] = mu
        stds[cls_id]  = sigma
    return means, stds


class SemanticEmissivityLoss(nn.Module):
    """
    给定TeVNet预测的发射率图 e_pred [B,1,H,W]
    和语义分割mask M [B,H,W] (long, class indices)
    计算每个像素的发射率是否落在对应类别的物理合理范围内。

    损失为: mean over valid pixels of relu(|e_pred - mu| - margin * sigma)
    即只惩罚超出 (mu ± margin*sigma) 范围的预测，容忍范围内的偏差。
    """

    def __init__(self, margin: float = 1.5, unknown_class: int = -1):
        """
        margin: 允许几倍标准差的偏差才开始惩罚
        unknown_class: 标记为忽略的类别索引（如背景）
        """
        super().__init__()
        self.margin = margin
        self.unknown_class = unknown_class

    def forward(self, e_pred: torch.Tensor,
                seg_mask: torch.Tensor) -> torch.Tensor:
        """
        e_pred:   [B, 1, H, W], 值域 [0,1], 由 sigmoid 激活
        seg_mask: [B, H, W], long tensor, 类别索引
        """
        device = e_pred.device
        prior_means, prior_stds = build_prior_tensors(device)

        B, _, H, W = e_pred.shape
        e = e_pred.squeeze(1)  # [B, H, W]

        # 从先验表中 gather 每个像素对应的 mu 和 sigma
        flat_mask = seg_mask.view(-1).clamp(0, NUM_CLASSES - 1)  # [B*H*W]
        mu_flat    = prior_means[flat_mask].view(B, H, W)
        sigma_flat = prior_stds[flat_mask].view(B, H, W)

        # 计算偏差，只惩罚超出 margin*sigma 的部分
        deviation = torch.abs(e - mu_flat)
        threshold = self.margin * sigma_flat
        penalty = F.relu(deviation - threshold)  # [B, H, W]

        # 排除未知类别的像素
        valid_mask = (seg_mask != self.unknown_class).float()
        # 也排除先验表中不确定的像素（std很大说明先验弱）
        confident_mask = (sigma_flat < 0.12).float()
        final_mask = valid_mask * confident_mask

        if final_mask.sum() < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = (penalty * final_mask).sum() / final_mask.sum()
        return loss
