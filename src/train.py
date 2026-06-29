"""
Training script for Multi-Scale MIL with LEM regularization.

FIXES applied (plan #5, #7, #8):
|- AMP: autocast + GradScaler for forward pass; LEM always computed in fp32.
|- LEM disabled during validation (compute_lem=False).
|- TensorBoard scalar LEM/active distinguishes "not yet active" from "zero".
|- Early stopping with patience.
|- Weighted sampler for class imbalance.
|- Checkpoint resume correctly restores current_epoch.
|- num_workers capped at 2 for Windows safety.
"""
import os
import sys
import time
import traceback

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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:  # wandb is optional; training still works without it
    wandb = None
    _WANDB_AVAILABLE = False

from src.config import load_config
from src.data.dataset import MammoMultiScaleDataset
from src.models.multi_scale_mil import MultiScaleMIL
from src.losses.total_loss import TotalLoss


# ---- pipeline stage logger --------------------------------------------------
_T0 = time.time()


def _ts() -> str:
    """Elapsed seconds since process start, formatted as [t=12.3s]."""
    return f"[t={time.time() - _T0:7.2f}s]"


def stage(msg: str) -> None:
    """Print a stage checkpoint with timestamp; always flushed."""
    print(f"{_ts()} [stage] {msg}", flush=True)


def stage_err(msg: str) -> None:
    """Print an error checkpoint; always flushed."""
    print(f"{_ts()} [stage] ERROR: {msg}", flush=True)


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


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler, gradient_clip=1.0,
                wandb_run=None, wandb_log_interval: int = 0, global_step_ref: list = None):
    model.train()
    total_loss, total_cls, total_lem = 0.0, 0.0, 0.0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # batch['patches']['scale_*'] shape: [B, N, 1, H, W]. The model handles
        # batching by reshaping to [B*N, ...] for the encoder and [B, N, D] for MIL.
        patches_dict = {
            'scale_0': batch['patches']['scale_0'].to(device),
            'scale_1': batch['patches']['scale_1'].to(device),
            'scale_2': batch['patches']['scale_2'].to(device),
        }
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        with autocast(enabled=scaler.is_enabled()):
            outputs = model(patches_dict, return_feature_map=True)
            logits = outputs['logits']
            feature_map = outputs['feature_map']
            loss, loss_dict = loss_fn(
                logits, labels, feature_map, compute_lem=True
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cls += loss_dict['cls_loss']
        total_lem += loss_dict['lem_loss']

        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Cls': f"{loss_dict['cls_loss']:.4f}",
            'LEM': f"{loss_dict['lem_loss']:.4f}",
        })

        # Optional: per-batch wandb logging
        if wandb_run is not None and wandb_log_interval > 0 and global_step_ref is not None:
            if global_step_ref[0] % wandb_log_interval == 0:
                wandb_run.log({
                    'train/batch/loss': loss.item(),
                    'train/batch/cls_loss': loss_dict['cls_loss'],
                    'train/batch/lem_loss': loss_dict['lem_loss'],
                    'train/batch/lr': optimizer.param_groups[0]['lr'],
                    'train/step': global_step_ref[0],
                }, step=global_step_ref[0])
            global_step_ref[0] += 1

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
            patches_dict = {
                'scale_0': batch['patches']['scale_0'].to(device),
                'scale_1': batch['patches']['scale_1'].to(device),
                'scale_2': batch['patches']['scale_2'].to(device),
            }
            labels = batch['label'].to(device)

            with autocast(enabled=scaler.is_enabled()):
                outputs = model(patches_dict, return_feature_map=True)
                logits = outputs['logits']
                feature_map = outputs['feature_map']
                # FIXED (plan #7): Disable LEM during validation
                loss, loss_dict = loss_fn(
                    logits, labels, feature_map, compute_lem=False
                )

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels).ravel()

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'probs': all_probs,
        'labels': all_labels,
    }


def main(config_path: str, experiment_dir: str, resume_path: str = None, max_epochs: int = None):
    stage(f"start  pid={os.getpid()}  config={config_path}  exp_dir={experiment_dir}")

    stage("set_seed(42)")
    set_seed(42)

    stage("loading config ...")
    config = load_config(config_path)
    stage(f"config loaded: data.root_dir={config.data.root_dir}  "
          f"training.epochs={config.training.epochs}  batch_size={config.training.batch_size}  "
          f"use_amp={config.training.use_amp}")

    if max_epochs is not None:
        config.training.epochs = max_epochs
        stage(f"overrode epochs -> {config.training.epochs}")

    stage(f"mkdir {experiment_dir}")
    os.makedirs(experiment_dir, exist_ok=True)
    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(experiment_dir, 'logs')
    writer = SummaryWriter(log_dir)
    stage(f"SummaryWriter -> {log_dir}")

    # CPU/GPU selection. Set ADDNET_FORCE_CPU=1 to bypass CUDA (useful when the
    # installed PyTorch does not support the current GPU's compute capability,
    # which causes hard process termination with no traceback).
    force_cpu = os.environ.get("ADDNET_FORCE_CPU", "").strip().lower() in ("1", "true", "yes")
    use_amp = config.training.use_amp and not force_cpu
    if force_cpu:
        device = torch.device('cpu')
        stage("device: cpu (forced via ADDNET_FORCE_CPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        try:
            cap = torch.cuda.get_device_capability(0)
            gpu_name = torch.cuda.get_device_name(0)
            stage(f"device: cuda  gpu={gpu_name}  sm_{cap[0]}{cap[1]}")
            if cap[0] >= 12:
                stage("WARNING: Blackwell-class GPU (sm_120+). "
                      "PyTorch <2.7+cu128 will crash on first CUDA op. "
                      "Re-run with ADDNET_FORCE_CPU=1 if that happens.")
        except Exception as e:
            stage_err(f"failed to query CUDA device: {e}")
            raise
    else:
        device = torch.device('cpu')
        stage("device: cpu (no CUDA available)")

    stage(f"loading train dataset from {config.data.root_dir} ...")
    t = time.time()
    train_dataset = MammoMultiScaleDataset(
        root_dir=config.data.root_dir, split='train', config=config
    )
    stage(f"train dataset loaded: {len(train_dataset)} samples  ({time.time() - t:.1f}s)")

    stage(f"loading val dataset from {config.data.root_dir} ...")
    t = time.time()
    val_dataset = MammoMultiScaleDataset(
        root_dir=config.data.root_dir, split='val', config=config
    )
    stage(f"val dataset loaded: {len(val_dataset)} samples  ({time.time() - t:.1f}s)")

    if config.training.use_weighted_sampler:
        stage("building weighted sampler ...")
        train_sampler = get_sampler(train_dataset)
        shuffle = False
    else:
        train_sampler, shuffle = None, True

    stage(f"building DataLoaders (num_workers=0, pin_memory=True) ...")
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, sampler=train_sampler, shuffle=shuffle,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    stage(f"DataLoaders: train={len(train_loader)} batches, val={len(val_loader)} batches")

    stage(f"building MultiScaleMIL model on {device} ...")
    t = time.time()
    try:
        model = MultiScaleMIL(config).to(device)
    except Exception as e:
        stage_err(f"MultiScaleMIL(...) failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stage(f"model built: {n_trainable:,} trainable / {n_params:,} total params  "
          f"({time.time() - t:.1f}s)")

    stage("building optimizer ...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=config.training.scheduler_patience,
        factor=config.training.scheduler_factor,
    )
    scaler = GradScaler(enabled=use_amp)
    stage(f"optimizer=Adam  scaler.amp_enabled={scaler.is_enabled()}")

    stage("building loss fn ...")
    loss_fn = TotalLoss(
        config=config,
        warmup_epoch=config.training.warmup_epochs,
    )
    stage("loss fn ready")

    # ---- Weights & Biases init (optional; never fatal) -------------------
    wandb_cfg = getattr(config.logging, 'wandb', None)
    wandb_enabled = bool(wandb_cfg and wandb_cfg.get('enabled', False))
    # CLI override (--wandb / --no_wandb)
    _override = os.environ.get('ADDNET_WANDB_OVERRIDE', '').strip()
    if _override == '1':
        wandb_enabled = True
    elif _override == '0':
        wandb_enabled = False
    wandb_run = None
    if wandb_enabled and not _WANDB_AVAILABLE:
        stage("WARNING: wandb requested but not installed (pip install wandb). Continuing without it.")
        wandb_enabled = False
    if wandb_enabled:
        try:
            run_name = wandb_cfg.get('name') or os.path.basename(os.path.normpath(experiment_dir))
            wandb_run = wandb.init(
                project=wandb_cfg.get('project', 'add-net'),
                entity=wandb_cfg.get('entity') or None,
                name=run_name,
                mode=wandb_cfg.get('mode', 'online'),
                config={
                    'data': dict(config.data),
                    'model': dict(config.model),
                    'loss': dict(config.loss),
                    'training': dict(config.training),
                    'logging': dict(config.logging),
                    'experiment_dir': experiment_dir,
                    'force_cpu': force_cpu,
                },
                tags=list(wandb_cfg.get('tags', [])) or None,
                dir=experiment_dir,
                resume='allow',
            )
            stage(f"wandb initialized: project={wandb_cfg.get('project')}  "
                  f"run_name={run_name}  url={wandb_run.url}")
            if wandb_cfg.get('log_gradients', False):
                try:
                    wandb.watch(
                        model,
                        log=wandb_cfg.get('watch_log', 'gradients'),
                        log_freq=wandb_cfg.get('watch_freq', 200),
                    )
                    stage(f"wandb.watch enabled: log={wandb_cfg.get('watch_log')} "
                          f"freq={wandb_cfg.get('watch_freq')}")
                except Exception as e:
                    stage_err(f"wandb.watch failed: {e}")
        except Exception as e:
            stage_err(f"wandb.init failed: {e}  (continuing without wandb)")
            wandb_run = None

    # ---- runtime state ----------------------------------------------------
    start_epoch = 0
    best_val_auc = 0.0
    patience_counter = 0
    global_step = [0]  # mutable counter for batch-level wandb logging
    wandb_log_interval = int((wandb_cfg or {}).get('log_interval', 0)) if wandb_run else 0

    if resume_path and os.path.exists(resume_path):
        stage(f"resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_auc = ckpt.get('val_auc', 0.0)
        loss_fn.set_epoch(start_epoch)
        stage(f"resumed epoch {start_epoch}, best AUC {best_val_auc:.4f}")
    else:
        stage(f"starting fresh from epoch 0 (target epochs={config.training.epochs})")

    stage("entering training loop")
    for epoch in range(start_epoch, config.training.epochs):
        loss_fn.set_epoch(epoch)
        print(f"\n{_ts()} Epoch {epoch+1}/{config.training.epochs}", flush=True)
        print("-" * 40, flush=True)

        stage(f"epoch {epoch+1}: train phase")
        t_train = time.time()
        try:
            train_stats = train_epoch(
                model, train_loader, optimizer, loss_fn, device, scaler,
                gradient_clip=config.training.gradient_clip,
                wandb_run=wandb_run,
                wandb_log_interval=wandb_log_interval,
                global_step_ref=global_step,
            )
        except Exception as e:
            stage_err(f"train_epoch failed at epoch {epoch+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise
        train_sec = time.time() - t_train
        stage(f"epoch {epoch+1}: train done  "
              f"loss={train_stats['loss']:.4f}  cls={train_stats['cls_loss']:.4f}  "
              f"lem={train_stats['lem_loss']:.4f}  ({train_sec:.1f}s)")

        stage(f"epoch {epoch+1}: validation phase")
        t_val = time.time()
        try:
            val_stats = validate(model, val_loader, loss_fn, device, scaler)
        except Exception as e:
            stage_err(f"validate failed at epoch {epoch+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise
        val_sec = time.time() - t_val
        stage(f"epoch {epoch+1}: val done  loss={val_stats['loss']:.4f}  "
              f"({val_sec:.1f}s)")

        val_auc = roc_auc_score(val_stats['labels'], val_stats['probs'])
        val_preds = (val_stats['probs'] >= 0.5).astype(int)
        val_precision = precision_score(val_stats['labels'], val_preds, zero_division=0)
        val_recall = recall_score(val_stats['labels'], val_preds, zero_division=0)
        val_f1 = f1_score(val_stats['labels'], val_preds, zero_division=0)
        scheduler.step(val_stats['loss'])

        lem_active = 1.0 if epoch >= config.training.warmup_epochs else 0.0

        writer.add_scalar('Loss/Train', train_stats['loss'], epoch)
        writer.add_scalar('Loss/Val', val_stats['loss'], epoch)
        writer.add_scalar('AUC/Val', val_auc, epoch)
        writer.add_scalar('Precision/Val', val_precision, epoch)
        writer.add_scalar('Recall/Val', val_recall, epoch)
        writer.add_scalar('F1/Val', val_f1, epoch)
        writer.add_scalar('Cls/Train', train_stats['cls_loss'], epoch)
        writer.add_scalar('LEM/Train', train_stats['lem_loss'], epoch)
        writer.add_scalar('LEM/active', lem_active, epoch)

        epoch_log = {
            'train/loss': train_stats['loss'],
            'train/cls_loss': train_stats['cls_loss'],
            'train/lem_loss': train_stats['lem_loss'],
            'train/lem_active': lem_active,
            'val/loss': val_stats['loss'],
            'val/auc': val_auc,
            'val/precision': val_precision,
            'val/recall': val_recall,
            'val/f1': val_f1,
            'val/pos_rate': float(val_stats['labels'].mean()),
            'optim/lr': optimizer.param_groups[0]['lr'],
            'optim/scaler_scale': float(scaler.get_scale()),
            'time/train_sec': train_sec,
            'time/val_sec': val_sec,
        }
        if device.type == 'cuda':
            try:
                torch.cuda.synchronize()
                epoch_log['sys/gpu_mem_alloc_gb'] = torch.cuda.memory_allocated() / 1e9
                epoch_log['sys/gpu_mem_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            except Exception:
                pass

        if wandb_run is not None:
            try:
                wandb_run.log(epoch_log, step=epoch)
            except Exception as e:
                stage_err(f"wandb.log failed at epoch {epoch+1}: {e}")

        print(f"Train: Loss={train_stats['loss']:.4f}  "
              f"Cls={train_stats['cls_loss']:.4f}  "
              f"LEM={train_stats['lem_loss']:.4f}", flush=True)
        print(f"Val:   Loss={val_stats['loss']:.4f}  AUC={val_auc:.4f}  "
              f"Precision={val_precision:.4f}  Recall={val_recall:.4f}  "
              f"F1={val_f1:.4f}", flush=True)

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
            print(f"  -> Saved best model (AUC: {val_auc:.4f})", flush=True)
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{config.training.early_stopping_patience})", flush=True)

        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}", flush=True)
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
    print(f"\n{_ts()} Training complete. Best val AUC: {best_val_auc:.4f}", flush=True)
    if wandb_run is not None:
        try:
            wandb_run.summary['best_val_auc'] = best_val_auc
            wandb_run.finish()
            stage("wandb.run.finish()")
        except Exception as e:
            stage_err(f"wandb.finish failed: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--exp_dir', type=str, default='./experiments/exp_001')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Override training.epochs from the config (for smoke tests).')
    parser.add_argument('--wandb', dest='wandb', action='store_true',
                        help='Enable Weights & Biases logging (overrides config).')
    parser.add_argument('--no_wandb', dest='wandb', action='store_false',
                        help='Disable Weights & Biases logging (overrides config).')
    parser.set_defaults(wandb=None)
    args = parser.parse_args()

    # Allow CLI flag to override the config's wandb.enabled flag before main()
    if args.wandb is not None:
        os.environ['ADDNET_WANDB_OVERRIDE'] = '1' if args.wandb else '0'

    try:
        main(args.config, args.exp_dir, args.resume, max_epochs=args.max_epochs)
    except SystemExit as e:
        # argparse / sys.exit - normal flow
        raise
    except BaseException as e:
        # Any other error: print a single, clear traceback and exit non-zero so
        # the shell shows the failure instead of returning to the prompt.
        stage_err(f"unhandled {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
