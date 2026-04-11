import argparse
import os
import copy
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from models import TeVNet
from utils import AverageMeter, TeVloss


# ---------------------------------------------------------------------------
# 数据集：直接读文件夹，不依赖任何 txt 文件
# ---------------------------------------------------------------------------

def build_transforms(is_train: bool, size: int = 512):
    """返回训练或验证用的图像变换流水线。"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size, padding=size // 8),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])


class FolderDataset(Dataset):
    """
    直接扫描 img_dir 文件夹中的所有图片，不依赖 txt 文件。
    支持 jpg / png / jpeg / bmp / tif 格式。
    self-supervised 模式：输入和标签是同一张图片（与原 TeVNet 一致）。
    """

    EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    def __init__(self, img_dir: str, is_train: bool = True,
                 size: int = 512):
        super().__init__()
        if not os.path.isdir(img_dir):
            raise ValueError(f"img_dir 不存在或不是文件夹: {img_dir}")

        self.transform = build_transforms(is_train, size)
        self.images = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ])

        if len(self.images) == 0:
            raise RuntimeError(
                f"在 {img_dir} 下没有找到任何图片，"
                f"支持格式: {self.EXTENSIONS}"
            )
        print(f"[FolderDataset] {'train' if is_train else 'val'} "
              f"共找到 {len(self.images)} 张图片，路径: {img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        tensor = self.transform(img)
        # self-supervised: 输入 == 标签
        return tensor, tensor


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def save_json(data, path):
    with open(path, mode="w") as f:
        json.dump(data, f, indent=4)


def save_model_checkpoint(state_dict, path):
    torch.save({'state_dict': state_dict}, path)


def save_loss_to_file(loss, epoch, path):
    with open(path, 'a+') as f:
        f.write(f'epoch: {epoch}; loss: {loss:.8f}\n')


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.outputs_dir, exist_ok=True)
    save_json(vars(args), os.path.join(args.outputs_dir, "params.json"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # 模型：TeVNet（原始，不含 seg_head）
    # seg_head 集成在 HADARNet 里，TeVNet 只在 Stage-1 独立训练
    # ------------------------------------------------------------------
    model = TeVNet(
        in_channels=3,
        out_channels=2 + args.vnums,
        args=args
    ).to(device)
    model = nn.DataParallel(model, device_ids=[0])

    # ------------------------------------------------------------------
    # 语义发射率损失（核心改动）
    # ------------------------------------------------------------------
    use_emiss_loss = args.emiss_loss_weight > 0 and args.use_seg_head

    if use_emiss_loss:
        # 动态导入，避免环境中没有安装时报错
        try:
            from ldm.modules.HADARNet.semantic_emissivity import (
                SemanticEmissivityLoss,
                NUM_CLASSES,
            )
            emiss_loss_fn = SemanticEmissivityLoss(margin=1.5).to(device)

            # 轻量语义头：接在 TeVNet encoder 最后特征图之后
            # resnet18 最后一层输出 512 通道
            encoder_out_ch = {
                'resnet18': 512, 'resnet34': 512,
                'resnet50': 2048, 'resnet101': 2048,
            }.get(args.smp_encoder, 512)

            seg_head = nn.Sequential(
                nn.Conv2d(encoder_out_ch, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, NUM_CLASSES, 1),
            ).to(device)
            seg_optimizer = optim.Adam(seg_head.parameters(), lr=args.lr)
            print(f"[SemanticEmissivity] 启用语义发射率损失, "
                  f"weight={args.emiss_loss_weight}, "
                  f"num_classes={NUM_CLASSES}")
        except ImportError as e:
            print(f"[警告] 无法导入 SemanticEmissivityLoss，跳过: {e}")
            use_emiss_loss = False
    else:
        seg_head = None
        emiss_loss_fn = None
        seg_optimizer = None

    # ------------------------------------------------------------------
    # 断点续训
    # ------------------------------------------------------------------
    if args.resume:
        ckpt = torch.load(
            args.resume,
            map_location=lambda storage, loc: storage
        )
        state_dict = model.state_dict()
        for n, p in ckpt['state_dict'].items():
            if n in state_dict:
                state_dict[n].copy_(p)
            else:
                print(f"[Resume] 跳过未知键: {n}")
        print(f"[Resume] 从 {args.resume} 恢复权重")

    # ------------------------------------------------------------------
    # 损失函数 & 优化器
    # ------------------------------------------------------------------
    lossmodule = TeVloss(vnums=args.vnums)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 数据加载（直接读文件夹，不需要 txt）
    # ------------------------------------------------------------------
    train_dataset = FolderDataset(
        img_dir=args.train_dir,
        is_train=True,
        size=args.img_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    eval_dataset = FolderDataset(
        img_dir=args.eval_dir,
        is_train=False,
        size=args.img_size,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 训练主循环（单一循环，合并了原始重建损失 + 语义发射率损失）
    # ------------------------------------------------------------------
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = float('inf')

    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        if seg_head is not None:
            seg_head.train()

        epoch_losses = AverageMeter()
        epoch_rec_losses = AverageMeter()
        epoch_emiss_losses = AverageMeter()

        # ---------- 定期保存检查点 ----------
        if epoch % args.num_epochs_save == 0 or epoch == args.num_epochs - 1:
            save_model_checkpoint(
                model.state_dict(),
                os.path.join(args.outputs_dir, f'epoch_{epoch}.pth')
            )

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch}/{args.num_epochs - 1}') as t:

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # ---- 前向传播 ----
                # TeVNet 本身的输出（TeV 分解）
                # model 是 DataParallel，module 才是真正的 TeVNet
                preds = model(inputs)  # [B, 2+vnums, H, W]

                # ---- 重建损失（原有）----
                rec_loss = lossmodule.loss_rec(preds, labels)

                # ---- 语义发射率先验损失（新增）----
                if use_emiss_loss and seg_head is not None:
                    # 从 TeVNet encoder 取最后一层特征
                    # tevnet 是 smp 的 UNet，encoder 是独立子模块
                    inner_model = (model.module
                                   if isinstance(model, nn.DataParallel)
                                   else model)
                    with torch.no_grad():
                        enc_feats = inner_model.tevnet.encoder(inputs)
                    last_feat = enc_feats[-1]  # [B, C, H/32, W/32]

                    seg_logits = seg_head(last_feat)
                    import torch.nn.functional as F
                    seg_logits = F.interpolate(
                        seg_logits,
                        size=inputs.shape[2:],
                        mode='bilinear',
                        align_corners=False,
                    )  # [B, num_classes, H, W]

                    seg_mask = seg_logits.argmax(dim=1)  # [B, H, W]
                    e_pred = preds[:, 0:1, :, :]          # emissivity

                    emiss_loss = emiss_loss_fn(e_pred, seg_mask)
                    loss = rec_loss + args.emiss_loss_weight * emiss_loss

                    epoch_emiss_losses.update(emiss_loss.item(), len(inputs))
                else:
                    loss = rec_loss
                    emiss_loss = torch.tensor(0.0)

                epoch_rec_losses.update(rec_loss.item(), len(inputs))
                epoch_losses.update(loss.item(), len(inputs))

                # ---- 反向传播 ----
                optimizer.zero_grad()
                if seg_optimizer is not None:
                    seg_optimizer.zero_grad()

                loss.backward()

                optimizer.step()
                if seg_optimizer is not None:
                    seg_optimizer.step()

                t.set_postfix(
                    loss=f'{epoch_losses.avg:.6f}',
                    rec=f'{epoch_rec_losses.avg:.6f}',
                    emiss=f'{epoch_emiss_losses.avg:.6f}',
                )
                t.update(len(inputs))

        # ---------- 保存本轮 loss ----------
        save_loss_to_file(
            epoch_losses.avg, epoch,
            os.path.join(args.outputs_dir, 'loss.txt')
        )
        save_model_checkpoint(
            model.state_dict(),
            os.path.join(args.outputs_dir, 'last.pth')
        )

        # ------------------------------------------------------------------
        # 验证
        # ------------------------------------------------------------------
        if epoch % args.num_epochs_val == 0 or epoch == args.num_epochs - 1:
            model.eval()
            if seg_head is not None:
                seg_head.eval()
            epoch_val_loss = AverageMeter()

            with torch.no_grad():
                for inputs, labels in eval_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    preds = model(inputs)
                    val_loss = lossmodule.loss_rec(preds, labels)
                    epoch_val_loss.update(val_loss.item(), len(inputs))

            print(f'\n[Val] Epoch {epoch} | val_loss: {epoch_val_loss.avg:.8f}')
            save_loss_to_file(
                epoch_val_loss.avg, epoch,
                os.path.join(args.outputs_dir, 'val_loss.txt')
            )

            if epoch_val_loss.avg < best_loss:
                best_loss = epoch_val_loss.avg
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())
                save_model_checkpoint(
                    best_weights,
                    os.path.join(args.outputs_dir, 'best.pth')
                )
                print(f'[Val] 更新最优模型，epoch={best_epoch}, '
                      f'val_loss={best_loss:.8f}')

    # ------------------------------------------------------------------
    # 训练结束
    # ------------------------------------------------------------------
    print(f'\n训练完成。Best epoch: {best_epoch}, '
          f'Best val_loss: {best_loss:.4f}')
    save_loss_to_file(
        best_loss, best_epoch,
        os.path.join(args.outputs_dir, 'val_loss.txt')
    )
    # 最终保存一次最优权重（防止循环里没进 if）
    save_model_checkpoint(
        best_weights,
        os.path.join(args.outputs_dir, 'best.pth')
    )


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TeVNet Stage-1 训练（语义发射率先验版）'
    )

    # --- 数据路径（直接填文件夹，不需要 txt）---
    parser.add_argument(
        '--train-dir', type=str, required=True,
        help='训练集图片文件夹，例如 /data/KAIST/train/infrared'
    )
    parser.add_argument(
        '--eval-dir', type=str, required=True,
        help='验证集图片文件夹，例如 /data/KAIST/val/infrared'
    )
    parser.add_argument(
        '--img-size', type=int, default=512,
        help='训练时统一 resize 的图片尺寸，默认 512'
    )

    # --- 输出 ---
    parser.add_argument(
        '--outputs-dir', type=str, required=True,
        help='模型和日志的保存目录'
    )

    # --- 训练超参 ---
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--batch-size',     type=int,   default=64)
    parser.add_argument('--num-epochs',     type=int,   default=1000)
    parser.add_argument('--num-epochs-save',type=int,   default=50)
    parser.add_argument('--num-epochs-val', type=int,   default=10)
    parser.add_argument('--num-workers',    type=int,   default=8)
    parser.add_argument('--resume',         type=str,   default=None)
    parser.add_argument('--start_epoch',    type=int,   default=0)

    # --- 模型结构 ---
    parser.add_argument('--vnums',               type=int,   default=4)
    parser.add_argument('--smp_model',           type=str,   default='Unet')
    parser.add_argument('--smp_encoder',         type=str,   default='resnet18')
    parser.add_argument('--smp_encoder_weights', type=str,   default='imagenet')

    # --- 语义发射率先验损失（新增）---
    parser.add_argument(
        '--emiss_loss_weight', type=float, default=0.1,
        help='语义发射率先验损失的权重，设为 0 则关闭该损失'
    )
    parser.add_argument(
        '--use_seg_head', action='store_true', default=True,
        help='是否启用语义分割头来计算发射率先验损失'
    )

    args = parser.parse_args()
    main(args)
