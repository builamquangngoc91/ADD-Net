"""
Training script for Multi-Scale MIL with LEM regularization.

FIXES applied (plan #5, #7, #8):
- AMP: autocast + GradScaler for forward pass; LEM always computed in fp32.
- LEM disabled during validation (compute_lem=False).
- TensorBoard scalar LEM/active distinguishes "not yet active" from "zero".
- Early stopping with patience.
- Weighted sampler for class imbalance.
- Checkpoint resume correctly restores current_epoch.
- num_workers capped at 2 for Windows safety.
"""
import os
import sys

# Ensure repo root is on sys.path so `python src/train.py` works without PYTHONPATH.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.data.dataset import MammoMultiScaleDataset
from src.models.multi_scale_mil import MultiScaleMIL
from src.losses.total_loss import TotalLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sampler(dataset) -> WeightedRandomSampler:
    """Create weighted sampler for class imbalance."""
    labels = np.asarray(dataset.get_all_labels())
    labels = labels.astype(np.int64)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler, gradient_clip=1.0):
    model.train()
    total_loss, total_cls, total_lem = 0.0, 0.0, 0.0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # batch['patches']['scale_*'] shape: [B, N, 1, H, W] where B=batch_size, N=patches/bag.
        # MIL semantics: process each bag independently and accumulate loss.
        batch_loss = 0.0
        batch_cls = 0.0
        batch_lem = 0.0
        B = batch['label'].shape[0]
        labels = batch['label'].to(device)

        for b in range(B):
            patches_dict = {
                'scale_0': batch['patches']['scale_0'][b].to(device),
                'scale_1': batch['patches']['scale_1'][b].to(device),
                'scale_2': batch['patches']['scale_2'][b].to(device),
            }
            label_b = labels[b:b+1]

            with autocast(enabled=scaler.is_enabled()):
                outputs = model(patches_dict, return_feature_map=True)
                logits = outputs['logits']
                feature_map = outputs['feature_map']
                loss_b, loss_dict = loss_fn(
                    logits, label_b, feature_map, compute_lem=True
                )

            loss_b = loss_b / B
            scaler.scale(loss_b).backward()
            batch_loss += loss_b.item()
            batch_cls += loss_dict['cls_loss'] / B
            batch_lem += loss_dict['lem_loss'] / B

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        pbar.set_postfix({
            'Loss': f"{batch_loss:.4f}",
            'Cls': f"{batch_cls:.4f}",
            'LEM': f"{batch_lem:.4f}",
        })

        total_loss += batch_loss
        total_cls += batch_cls
        total_lem += batch_lem

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'cls_loss': total_cls / n,
        'lem_loss': total_lem / n,
    }


def validate(model, dataloader, loss_fn, device, scaler):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            labels = batch['label'].to(device)
            B = labels.shape[0]

            for b in range(B):
                patches_dict = {
                    'scale_0': batch['patches']['scale_0'][b].to(device),
                    'scale_1': batch['patches']['scale_1'][b].to(device),
                    'scale_2': batch['patches']['scale_2'][b].to(device),
                }
                label_b = labels[b:b+1]

                with autocast(enabled=scaler.is_enabled()):
                    outputs = model(patches_dict, return_feature_map=True)
                    logits = outputs['logits']
                    feature_map = outputs['feature_map']
                    # FIXED (plan #7): Disable LEM during validation
                    loss, loss_dict = loss_fn(
                        logits, label_b, feature_map, compute_lem=False
                    )

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=0)[1].cpu().numpy()
                all_probs.append(probs)
                all_labels.append(label_b.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).ravel()

    n = max(1, len(all_labels))
    return {
        'loss': total_loss / n,
        'probs': all_probs,
        'labels': all_labels,
    }


def main(config_path: str, experiment_dir: str, resume_path: str = None, max_epochs: int = None):
    set_seed(42)
    config = load_config(config_path)
    if max_epochs is not None:
        config.training.epochs = max_epochs

    os.makedirs(experiment_dir, exist_ok=True)
    ckpt_dir = os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    log_dir = os.path.join(experiment_dir, 'logs')
    writer = SummaryWriter(log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = MammoMultiScaleDataset(
        root_dir=config.data.root_dir, split='train', config=config
    )
    val_dataset = MammoMultiScaleDataset(
        root_dir=config.data.root_dir, split='val', config=config
    )

    if config.training.use_weighted_sampler:
        train_sampler = get_sampler(train_dataset)
        shuffle = False
    else:
        train_sampler, shuffle = None, True

    # FIXED (plan #3.7 from review): limit workers for Windows RAM safety
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, sampler=train_sampler, shuffle=shuffle,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = MultiScaleMIL(config).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=config.training.scheduler_patience,
        factor=config.training.scheduler_factor,
    )
    scaler = GradScaler(enabled=config.training.use_amp)

    loss_fn = TotalLoss(
        config=config,
        warmup_epoch=config.training.warmup_epochs,
    )

    # FIXED (plan #8): Track training state for resume
    start_epoch = 0
    best_val_auc = 0.0
    patience_counter = 0

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_auc = ckpt.get('val_auc', 0.0)
        loss_fn.set_epoch(start_epoch)
        print(f"Resumed epoch {start_epoch}, best AUC {best_val_auc:.4f}")

    for epoch in range(start_epoch, config.training.epochs):
        loss_fn.set_epoch(epoch)
        print(f"\nEpoch {epoch+1}/{config.training.epochs}")
        print("-" * 40)

        train_stats = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler,
            gradient_clip=config.training.gradient_clip,
        )
        val_stats = validate(model, val_loader, loss_fn, device, scaler)

        val_auc = roc_auc_score(val_stats['labels'], val_stats['probs'])
        scheduler.step(val_stats['loss'])

        lem_active = 1.0 if epoch >= config.training.warmup_epochs else 0.0

        writer.add_scalar('Loss/Train', train_stats['loss'], epoch)
        writer.add_scalar('Loss/Val', val_stats['loss'], epoch)
        writer.add_scalar('AUC/Val', val_auc, epoch)
        writer.add_scalar('Cls/Train', train_stats['cls_loss'], epoch)
        writer.add_scalar('LEM/Train', train_stats['lem_loss'], epoch)
        # FIXED (plan #7): separate scalar so flat warmup line is interpretable
        writer.add_scalar('LEM/active', lem_active, epoch)

        print(f"Train: Loss={train_stats['loss']:.4f}  "
              f"Cls={train_stats['cls_loss']:.4f}  "
              f"LEM={train_stats['lem_loss']:.4f}")
        print(f"Val:   Loss={val_stats['loss']:.4f}  AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_auc': val_auc,
                'config': config,
            }, os.path.join(experiment_dir, 'checkpoints', 'best_model.pth'))
            print(f"  -> Saved best model (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{config.training.early_stopping_patience})")

        # FIXED (plan #3.7): Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % config.logging.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_auc': val_auc,
                'config': config,
            }, os.path.join(experiment_dir, 'checkpoints', f'epoch_{epoch+1}.pth'))

    writer.close()
    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--exp_dir', type=str, default='./experiments/exp_001')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Override training.epochs from the config (for smoke tests).')
    args = parser.parse_args()
    main(args.config, args.exp_dir, args.resume, max_epochs=args.max_epochs)
