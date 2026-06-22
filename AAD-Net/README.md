# AAD-Net: Asymmetry-Aware Dual-View Network

A weakly supervised pipeline for breast cancer screening on 4-view mammography (L-CC, R-CC, L-MLO, R-MLO) using VinDr-MAMMO.

**Bi-Rads Label Mapping:**
- Bi-Rads 1, 2 → Normal (label = 0)
- Bi-Rads 4, 5 → Abnormal (label = 1)
- Bi-Rads 3 → **Strictly discarded**

---

## Project Structure

```
AAD-Net/
├── config/config.py              # All hyperparameters, paths, Bi-Rads rules
├── data_loaders/reader.py        # MammographyLoader (CSV parsing, Bi-Rads filtering)
├── stage1_alignment/             # Preprocessing: normalization + registration
│   ├── transforms.py             #   Otsu masking → Z-score → CLAHE
│   ├── registration.py           #   VoxelMorph deformable registration
│   ├── run_stage1.py             #   Run Stage 1
│   └── verify_stage1.py          #   Verify Stage 1 output
├── stage2_diffusion/             # Latent diffusion: learn normal breast prior
│   ├── autoencoder.py            #   Frozen VQ-GAN/KL-AE encoder-decoder
│   ├── unet_modules.py           #   Cross-Lateral Diffusion U-Net
│   ├── trainer.py                #   Noise-prediction diffusion loss
│   ├── run_stage2.py             #   Train diffusion model (Normal patients only)
│   └── verify_stage2.py          #   Verify U-Net integrity
├── stage3_pseudo_labeling/        # Pseudo-label generation
│   ├── inference.py              #   Bi-directional DDIM sampler (L→R, R→L)
│   ├── perceptual_loss.py        #   Med-LPIPS (VGG-16, cosine distance)
│   ├── statistical_gmm.py        #   GMM soft labeling with scaled sigmoid
│   ├── run_stage3.py             #   Generate pseudo_labels.npy
│   └── verify_stage3.py          #   Verify pseudo-label validity
├── stage4_classifier/            # Weakly supervised classifier
│   ├── patch_supcon.py           #   Supervised Contrastive Loss
│   ├── fusion_network.py         #   ResNet-50 patch extractor + adaptive fusion
│   ├── run_stage4.py             #   Teacher-Student training
│   ├── evaluator.py              #   AUC + Sensitivity@95% metrics
│   └── verify_stage4.py          #   Assert AUC > 0.5
├── runner/run_all.py              # Run all stages sequentially
├── processed/                    # Runtime output (aligned .pt files, manifests)
├── checkpoints/                  # Trained model checkpoints
├── weights/                     # Place pretrained weights here
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies:
- `torch`, `torchvision`
- `numpy`, `pandas`, `Pillow`
- `scikit-learn`
- `opencv-python`
- `scipy`

---

## Pre-trained Weights (Optional)

Place downloaded weights into `AAD-Net/weights/`:

| File | Description |
|---|---|
| `vxm_breast.h5` | Pre-trained VoxelMorph breast registration model |
| `autoencoder.pth` | Pre-trained VQ-GAN or KL-Autoencoder |
| `radimagenet_vgg16.pth` | VGG-16 pretrained on RadImageNet |

Without these, the pipeline uses **dummy fallbacks** for the affected modules, allowing you to debug shapes and data flow immediately.

---

## Quick Start

### Run Everything

```bash
cd AAD-Net
python runner/run_all.py
```

Each stage runs sequentially. If any stage fails, the pipeline halts and reports the error.

### Run Stages Individually

**Stage 1 — Preprocessing (Normalization + Registration)**
```bash
python stage1_alignment/run_stage1.py
```
Output: `processed/stage1_aligned/*.pt` + `processed/manifest_stage1.json`

**Stage 2 — Diffusion Model Training (Normal patients only)**
```bash
python stage2_diffusion/run_stage2.py
```
Output: `checkpoints/diffusion_model.pth`

**Stage 3 — Pseudo-Label Generation**
```bash
python stage3_pseudo_labeling/run_stage3.py
```
Output: `processed/stage3_pseudolabels/pseudo_labels.npy`

**Stage 4 — Classifier Training + Evaluation**
```bash
python stage4_classifier/run_stage4.py
python stage4_classifier/verify_stage4.py
```
Output: `checkpoints/classifier_student.pth`

### Verify Any Stage

Each stage has a matching `verify_*.py` script:
```bash
python stage1_alignment/verify_stage1.py
python stage2_diffusion/verify_stage2.py
python stage3_pseudo_labeling/verify_stage3.py
python stage4_classifier/verify_stage4.py
```

---

## Key Hyperparameters (config/config.py)

| Parameter | Value | Description |
|---|---|---|
| `TARGET_SIZE` | `(1024, 512)` | Image resolution (W, H) |
| `PATCH_SIZE` | `128` | Patch size for LPIPS and SupCon |
| `STRIDE` | `64` | Patch extraction stride |
| `DIFF_EPOCHS` | `100` | Diffusion model training epochs |
| `CLS_EPOCHS` | `50` | Classifier training epochs |
| `LR` | `1e-4` | Learning rate |
| `ACCUMULATION_STEPS` | `4` | Gradient accumulation steps |
| `DDIM_STEPS` | `20` | DDIM sampling steps |
| `EMA_DECAY` | `0.999` | Teacher EMA decay |
