from torch import nn
import torch.nn.functional as F
import torch
import segmentation_models_pytorch as smp
# 新增导入
from ldm.modules.HADARNet.semantic_emissivity import SemanticEmissivityLoss


class HADARNet(nn.Module):
    def __init__(self,
                 smp_model,
                 smp_encoder,
                 in_channels=3,
                 out_channels=6,
                 ckpt_path=None,
                 ignore_keys=[],
                 # 新增参数
                 seg_num_classes=19,
                 use_seg_head=True
                 ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.use_seg_head = use_seg_head

        # 原始 TeV 分解网络
        self.tevnet = getattr(smp, smp_model)(
            encoder_name=smp_encoder,
            encoder_weights=None,
            in_channels=self.in_channels,
            classes=self.out_channels,
        )

        # 新增：轻量语义分割头，共享 encoder
        if self.use_seg_head:
            encoder_channels = self._get_encoder_out_channels(smp_encoder)
            self.seg_head = nn.Sequential(
                nn.Conv2d(encoder_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, seg_num_classes, kernel_size=1),
            )
            self.emiss_loss_fn = SemanticEmissivityLoss(margin=1.5)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def _get_encoder_out_channels(self, encoder_name: str) -> int:
        """根据 encoder 名称返回最后一层通道数"""
        channel_map = {
            'resnet18':  512,
            'resnet34':  512,
            'resnet50':  2048,
            'resnet101': 2048,
        }
        return channel_map.get(encoder_name, 512)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        for n, _ in list(sd.items()):
            sd[n.replace('module.', '')] = sd.pop(n)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        # strict=False 允许 seg_head 权重缺失（预训练模型没有这个头）
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        if missing:
            print(f"Missing Keys (expected for new seg_head): {missing}")

    def forward(self, x, rgb_for_seg=None):
        """
        x:            红外图像 [B, 3, H, W]，用于 TeV 分解
        rgb_for_seg:  可见光图像 [B, 3, H, W]，用于语义分割
                      如果为 None，则用 x 本身（仅推理时备用）
        返回:
            preds:     [B, out_channels, H, W]  TeV 预测
            seg_logits:[B, num_classes, H, W]   语义 logits（仅训练时）
        """
        # TeV 分解前向
        preds = self.tevnet(x)
        preds[:, 0, :, :] = torch.sigmoid(preds[:, 0, :, :])   # e ∈ [0,1]
        preds[:, 1, :, :] = F.relu(preds[:, 1, :, :])           # T ≥ 0

        if not self.use_seg_head or not self.training:
            return preds

        # 语义分割：用 RGB 图像过 encoder，取最后特征图
        seg_input = rgb_for_seg if rgb_for_seg is not None else x
        # 直接用 tevnet 的 encoder 做特征提取（共享权重）
        encoder_features = self.tevnet.encoder(seg_input)
        last_feat = encoder_features[-1]  # [B, C, H/32, W/32]

        seg_logits = self.seg_head(last_feat)
        # 上采样到原始分辨率
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:],
                                   mode='bilinear', align_corners=False)
        return preds, seg_logits

    def compute_emissivity_loss(self, preds, seg_logits):
        """
        在 TeVNet 训练循环中调用，计算发射率先验损失。
        preds:      [B, out_channels, H, W]
        seg_logits: [B, num_classes, H, W]
        """
        e_pred = preds[:, 0:1, :, :]          # emissivity channel
        seg_mask = seg_logits.argmax(dim=1)    # [B, H, W] pseudo label
        return self.emiss_loss_fn(e_pred, seg_mask)
