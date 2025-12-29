import argparse
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from copy import deepcopy
from tqdm import tqdm

from soccernet.calibration_data import LineKeypointDataset, LINE_CLASSES
from models.line_detection.model import KeyPointLineDetector, KeyPointLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


class Augmentation:
    """Strong augmentation for line detection."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, keypoints: torch.Tensor, visibility: torch.Tensor):
        if random.random() < self.p:
            image = torch.flip(image, dims=[-1])
            keypoints = keypoints.clone()
            keypoints[..., 0] = 1.0 - keypoints[..., 0]

        if random.random() < self.p:
            brightness = random.uniform(0.7, 1.3)
            image = image * brightness
            image = image.clamp(0, 1)

        if random.random() < self.p:
            contrast = random.uniform(0.7, 1.3)
            mean = image.mean()
            image = (image - mean) * contrast + mean
            image = image.clamp(0, 1)

        if random.random() < self.p:
            saturation = random.uniform(0.7, 1.3)
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            image = image * saturation + gray.unsqueeze(0) * (1 - saturation)
            image = image.clamp(0, 1)

        if random.random() < self.p * 0.3:
            noise = torch.randn_like(image) * 0.02
            image = (image + noise).clamp(0, 1)

        return image, keypoints, visibility


class MultiScaleDataset:
    """Wrapper for multi-scale training."""

    def __init__(self, dataset, scales: list = None):
        self.dataset = dataset
        self.scales = scales or [0.8, 0.9, 1.0, 1.1, 1.2]
        self.current_scale = 1.0

    def set_scale(self, scale: float):
        self.current_scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if self.current_scale != 1.0:
            image = sample["image"]
            _, h, w = image.shape
            new_h = int(h * self.current_scale)
            new_w = int(w * self.current_scale)
            image = F.interpolate(
                image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0)
            sample["image"] = image

        return sample


def get_lr_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 5, min_lr: float = 1e-6):
    """Warmup + Cosine Annealing scheduler."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return min_lr / optimizer.defaults["lr"] + (1 - min_lr / optimizer.defaults["lr"]) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    augmentation: Augmentation = None,
    ema: EMA = None,
) -> dict:
    model.train()
    total_loss = 0.0
    coord_loss = 0.0
    vis_loss = 0.0
    aux_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training")
    for i, batch in enumerate(pbar):
        images = batch["image"].to(device)
        gt_keypoints = batch["keypoints"].to(device)
        gt_visibility = batch["visibility"].to(device)
        gt_heatmaps = batch.get("heatmaps")
        if gt_heatmaps is not None:
            gt_heatmaps = gt_heatmaps.to(device)

        if augmentation is not None:
            for j in range(images.shape[0]):
                images[j], gt_keypoints[j], gt_visibility[j] = augmentation(
                    images[j], gt_keypoints[j], gt_visibility[j]
                )

        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(images)
            losses = loss_fn(outputs, gt_keypoints, gt_visibility, gt_heatmaps)
            loss = losses["total"] / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if ema is not None:
                ema.update(model)

        total_loss += losses["total"].item()
        coord_loss += losses["coord_loss"].item()
        vis_loss += losses["vis_loss"].item()
        aux_loss += losses.get("aux_loss", torch.tensor(0.0)).item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{losses['total'].item():.4f}",
            coord=f"{losses['coord_loss'].item():.4f}",
            vis=f"{losses['vis_loss'].item():.4f}",
        )

    return {
        "total": total_loss / n_batches,
        "coord": coord_loss / n_batches,
        "vis": vis_loss / n_batches,
        "aux": aux_loss / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    coord_loss = 0.0
    vis_loss = 0.0
    n_batches = 0

    total_visible = 0
    total_correct_5px = 0
    total_correct_10px = 0

    for batch in tqdm(loader, desc="Validation"):
        images = batch["image"].to(device)
        gt_keypoints = batch["keypoints"].to(device)
        gt_visibility = batch["visibility"].to(device)

        with autocast():
            outputs = model(images)
            losses = loss_fn(outputs, gt_keypoints, gt_visibility)

        total_loss += losses["total"].item()
        coord_loss += losses["coord_loss"].item()
        vis_loss += losses["vis_loss"].item()
        n_batches += 1

        pred_kp = outputs["keypoints"]

        vis_mask = gt_visibility > 0.5
        if vis_mask.sum() > 0:
            h, w = images.shape[-2:]
            pred_pixels = pred_kp.clone()
            pred_pixels[..., 0] *= w
            pred_pixels[..., 1] *= h
            gt_pixels = gt_keypoints.clone()
            gt_pixels[..., 0] *= w
            gt_pixels[..., 1] *= h

            dist = torch.sqrt(((pred_pixels - gt_pixels) ** 2).sum(dim=-1))
            dist_visible = dist[vis_mask]

            total_visible += vis_mask.sum().item()
            total_correct_5px += (dist_visible < 5).sum().item()
            total_correct_10px += (dist_visible < 10).sum().item()

    acc_5px = total_correct_5px / max(total_visible, 1) * 100
    acc_10px = total_correct_10px / max(total_visible, 1) * 100

    return {
        "total": total_loss / n_batches,
        "coord": coord_loss / n_batches,
        "vis": vis_loss / n_batches,
        "acc@5px": acc_5px,
        "acc@10px": acc_10px,
    }


def main():
    parser = argparse.ArgumentParser(description="KeyPoint Line Detection Training")
    parser.add_argument("--train-zip", type=str, default="soccernet/calibration/calibration/train.zip")
    parser.add_argument("--val-zip", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--output", type=str, default="keypoint_line_detector.pth")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--multi-scale", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = LineKeypointDataset(
        args.train_zip,
        width=args.width,
        height=args.height,
    )
    print(f"Train samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if args.val_zip:
        val_dataset = LineKeypointDataset(
            args.val_zip,
            width=args.width,
            height=args.height,
        )
        print(f"Val samples: {len(val_dataset)}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = KeyPointLineDetector(
        num_classes=len(LINE_CLASSES),
        max_points=12,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        pretrained=True,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    ema = EMA(model, decay=args.ema_decay)

    loss_fn = KeyPointLoss(
        coord_weight=5.0,
        vis_weight=1.0,
        heatmap_weight=0.5,
        aux_weight=0.5,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = get_lr_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler()

    augmentation = None if args.no_augmentation else Augmentation(p=0.5)

    best_val_loss = float("inf")
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 60)

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            accumulation_steps=args.accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            augmentation=augmentation,
            ema=ema,
        )
        print(
            f"Train | Loss: {train_metrics['total']:.4f} | "
            f"Coord: {train_metrics['coord']:.4f} | "
            f"Vis: {train_metrics['vis']:.4f} | "
            f"Aux: {train_metrics['aux']:.4f}"
        )

        if val_loader:
            val_metrics = validate(ema.shadow, val_loader, loss_fn, device)
            print(
                f"Val   | Loss: {val_metrics['total']:.4f} | "
                f"Coord: {val_metrics['coord']:.4f} | "
                f"Acc@5px: {val_metrics['acc@5px']:.1f}% | "
                f"Acc@10px: {val_metrics['acc@10px']:.1f}%"
            )

            improved = False
            if val_metrics["acc@5px"] > best_acc:
                best_acc = val_metrics["acc@5px"]
                improved = True

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                improved = True

            if improved:
                patience_counter = 0
                torch.save(ema.state_dict(), os.path.join(args.output_dir, args.output))
                print(f"  -> Saved best model (Acc@5px: {best_acc:.1f}%)")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping after {args.patience} epochs without improvement")
                    break
        else:
            torch.save(ema.state_dict(), os.path.join(args.output_dir, f"epoch_{epoch + 1}.pth"))

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_acc": best_acc,
                },
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
            )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Acc@5px: {best_acc:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
