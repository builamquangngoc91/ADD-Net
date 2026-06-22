To give your coding agent a bulletproof execution plan, we need to break the monolithic spec into **atomic, self-contained, ordered instructions**. 

Each instruction is a single step that produces a specific file. The agent must check off Step 1 before moving to Step 2. This guarantees modularity, easy debugging, and strict adherence to your Weakly Supervised (Bi-Rads 1-5, ignoring 3) pipeline.

Here is the **Step-by-Step Instruction Set for the Coding Agent**.

---

### 📋 INSTRUCTION SET FOR CODE AGENT

**Objective:** Build the AAD-Net pipeline. 
**Constraints:** Input is 4-view mammography (L-CC, R-CC, L-MLO, R-MLO). Bi-Rads 1,2 = Normal (Label 0); Bi-Rads 4,5 = Abnormal (Label 1); **Bi-Rads 3 must be completely discarded.**

---

#### Step 1: Create the Global Configuration (`config/config.py`)
**Goal:** Centralize all hyperparameters, dataset paths, and Bi-Rads mapping rules. 
**Inputs:** None.
**Output:** `AAD-Net/config/config.py`
**Action:**
*   Define `PROJECT_ROOT`, `RAW_DATA_DIR`, `PROCESSED_DIR`, `CHECKPOINT_DIR`.
*   Set `TARGET_SIZE = (1024, 512)`, `PATCH_SIZE = 128`, `STRIDE = 64`.
*   Define Bi-Rads rules: `NORMAL_BI_RADS = (1, 2)`, `ABNORMAL_BI_RADS = (4, 5)`, `IGNORE_BI_RADS = 3`.
*   Set Diffusion and Classifier training hyperparameters (`DIFF_EPOCHS`, `CLS_EPOCHS`, `LR`, `ACCUMULATION_STEPS`).
*   Set `DEVICE = "cuda"`.
**Verification:** The agent must successfully import `PipelineConfig()` in a Python interpreter without errors.

---

#### Step 2: Implement Data Loader with Bi-Rads Filtering (`data_loaders/reader.py`)
**Goal:** Read the CSV metadata, apply the weak supervision label mapping, load 4 images per patient, and resize to `1024x512`.
**Inputs:** `config/config.py`, CSV metadata path.
**Output:** `AAD-Net/data_loaders/reader.py`
**Action:**
*   Create a `MammographyLoader` class inheriting from `torch.utils.data.Dataset`.
*   In `__init__`: Iterate over the CSV. 
    *   If `bi_rads == 3`, `continue` (strictly discard).
    *   If `bi_rads in (1,2)`, assign `label = 0`.
    *   If `bi_rads in (4,5)`, assign `label = 1`.
*   In `_load_image`: Read image with `PIL`, convert to 'L' (grayscale), resize to `cfg.TARGET_SIZE` using `Image.BILINEAR`, convert to `torch.tensor`, normalize to `[0,1]`.
*   `__getitem__` must return a dictionary: `{'patient_id', 'L_CC', 'R_CC', 'L_MLO', 'R_MLO', 'label'}`.
**Verification:** Manually test `loader = MammographyLoader(cfg)`. Print `len(loader)` and check that the dataset size has decreased compared to the raw CSV (meaning Bi-Rads 3 were dropped).

---

#### Step 3: Implement Radiometric Normalization (`stage1_alignment/transforms.py`)
**Goal:** Reduce scanner variability and contrast differences. 
**Inputs:** A batch dictionary from Step 2.
**Output:** `AAD-Net/stage1_alignment/transforms.py`
**Action:**
*   Implement `run_radiometric_normalization(batch, target_size)`.
*   Inside this function:
    1.  Calculate Otsu threshold on each image to get a breast mask.
    2.  Apply `Z-score` normalization **only** on the tissue inside the mask (set background to 0).
    3.  Apply `CLAHE` (Contrast Limited Adaptive Histogram Equalization) to the masked area to enhance local tissue texture.
*   Return the normalized batch dictionary.
**Verification:** Run on a single batch. Save the output as an image using `torchvision.utils.save_image`. Visually check that contrast is enhanced and background is pure black.

---

#### Step 4: Implement Deformable Registration Wrapper (`stage1_alignment/registration.py`)
**Goal:** Align the Right breast (mirrored) to the Left breast using non-rigid registration.
**Inputs:** `voxelmorph` library.
**Output:** `AAD-Net/stage1_alignment/registration.py`
**Action:**
*   Create `run_deformable_registration(normalized_batch, reg_model)`.
*   Inside: 
    1.  Take `L_CC` and `R_CC`. Horizontally flip `R_CC` (using `torch.fliplr`).
    2.  Pass `L_CC` and flipped `R_CC` into the pre-trained `voxelmorph` model (loaded from `.h5`).
    3.  Apply the predicted dense displacement field to the flipped `R_CC` to get `R_CC_aligned`.
    4.  Repeat for MLO views.
    5.  Return the aligned batch dictionary.
**Verification:** Save `L_CC` and `R_CC_aligned` side-by-side. They should look geometrically synchronized (nipples and chest walls aligned).

---

#### Step 5: Implement Stage 1 Runner (`stage1_alignment/run_stage1.py`)
**Goal:** Orchestrate Steps 2, 3, and 4 to process the entire dataset.
**Inputs:** Step 2, 3, 4. Pre-trained `vxm_breast.h5`.
**Output:** `AAD-Net/processed/stage1_aligned/{patient_id}.pt` files & `manifest_stage1.json`.
**Action:**
*   Initialize `PipelineConfig` and `MammographyLoader`.
*   Load the pre-trained VoxelMorph model from Step 4.
*   Loop through the Data Loader. For each batch:
    1.  Apply `run_radiometric_normalization`.
    2.  Apply `run_deformable_registration`.
    3.  Save the aligned batch as a `.pt` file using `torch.save()`, with filename = `patient_id`.
    4.  Append `patient_id` to a list `saved_files`.
*   Save `manifest_stage1.json` containing `{"aligned_files": saved_files, "base_path": "..."}`.
**Verification:** Run the script. Check if `manifest_stage1.json` is created and `.pt` files exist.

---

#### Step 6: Implement Stage 1 Verifier (`stage1_alignment/verify_stage1.py`)
**Goal:** Sanity check data integrity before training.
**Inputs:** `manifest_stage1.json`.
**Output:** Console output (Success/Failure).
**Action:**
*   Load `manifest_stage1.json`.
*   Iterate through the first 5 files listed.
*   Load the `.pt` file. Run `assert` statements:
    *   `batch["L_CC"].shape == (1, 512, 1024)`
    *   `batch["R_CC"].shape == (1, 512, 1024)`
    *   `batch["label"] in [0, 1]` (strict binary check).
*   If all pass, print `✓ Stage 1 Data Verified.`
**Verification:** If the script prints the success message without raising `AssertionError`, proceed to Step 7.

---

#### Step 7: Implement Pre-trained Autoencoder Loader (`stage2_diffusion/autoencoder.py`)
**Goal:** Provide a frozen encoder/decoder to map images into Latent Space.
**Inputs:** Pretrained VQ-GAN/KL-AE weights.
**Output:** `AAD-Net/stage2_diffusion/autoencoder.py`
**Action:**
*   Implement `load_vqgan(cfg)`.
*   Load the weights.
*   Freeze all parameters: `for p in autoencoder.parameters(): p.requires_grad = False`.
*   Set model to `eval()` mode.
*   Return the model.
**Verification:** Run a dummy tensor `(1, 1, 512, 1024)` through the encoder. Ensure output latent shape is `(1, 4, 64, 128)`.

---

#### Step 8: Implement Cross-Lateral Diffusion U-Net (`stage2_diffusion/unet_modules.py`)
**Goal:** Create the U-Net backbone with Cross-Attention layers.
**Inputs:** None.
**Output:** `AAD-Net/stage2_diffusion/unet_modules.py`
**Action:**
*   Implement `CrossLateralDiffusionUNet`.
*   It must take `x` (Noisy latent), `t` (timestep), and `condition` (Latent of the contralateral breast).
*   Inside each downsample block, add a **Cross-Attention layer**.
    *   `Query` = Output of current block (Noisy side).
    *   `Key` and `Value` = Projected from the `condition` tensor.
*   Output shape must equal input shape `(Batch, Channels, 64, 128)`.
**Verification:** Pass random tensors `(1, 4, 64, 128)` into the model. Ensure there are no shape mismatch errors.

---

#### Step 9: Implement Diffusion Trainer Logic (`stage2_diffusion/trainer.py`)
**Goal:** Calculate the standard Diffusion Loss (Noise Prediction).
**Inputs:** Autoencoder, Diffusion U-Net.
**Output:** `AAD-Net/stage2_diffusion/trainer.py`
**Action:**
*   Implement `LatentDiffusionTrainer.train_step(L_cc, R_cc, autoencoder, diff_unet, timesteps)`.
*   **Logic:**
    1.  Encode `L_cc` into condition latent `z_L`.
    2.  Encode `R_cc` into target latent `z_R`.
    3.  Sample random timestep `t` and Gaussian noise `epsilon`.
    4.  Forward process: Add noise to `z_R` -> `z_R_t`.
    5.  Predict noise via `diff_unet(z_R_t, t, condition=z_L)`.
    6.  Compute MSE Loss: `loss = MSE(pred_noise, epsilon)`.
    7.  Return the loss.
**Verification:** Call this function once in a dummy test and verify `loss` is a scalar tensor with `requires_grad=True`.

---

#### Step 10: Implement Stage 2 Runner (`stage2_diffusion/run_stage2.py`)
**Goal:** Train the Diffusion U-Net on **Normal patients only** (Label=0).
**Inputs:** `manifest_stage1.json`, `autoencoder.py`, `trainer.py`.
**Output:** Checkpoint `AAD-Net/checkpoints/diffusion_model.pth`.
**Action:**
*   Load manifest. **Filter the `aligned_files` list to only those containing `_Normal`** in their ID (or check the label in the `.pt` file).
*   Load `autoencoder` and `diff_unet` to GPU.
*   Initialize `optim.Adam` for `diff_unet`.
*   Loop for `DIFF_EPOCHS`:
    *   Iterate through the filtered Normal files.
    *   Load `.pt`, extract `L_cc` and `R_cc`.
    *   Compute loss via Step 9.
    *   `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`.
    *   Log average loss per epoch.
*   Save `diff_unet.state_dict()` to `checkpoints/diffusion_model.pth`.
**Verification:** Checkpoint file is created. Run `verify_stage2.py` (Step 11) to ensure the model can be loaded.

---

#### Step 11: Implement Stage 2 Verifier (`stage2_diffusion/verify_stage2.py`)
**Goal:** Validate U-Net structural integrity.
**Action:**
*   Load `diffusion_model.pth` into `CrossLateralDiffusionUNet`.
*   Set to `eval()`.
*   Generate random dummy latents for noise and condition.
*   Run a forward pass. Ensure the output shape matches input shape.
*   Print `✓ Diffusion U-Net structural integrity verified.`

---

#### Step 12: Implement Bi-Directional DDIM Sampler (`stage3_pseudo_labeling/inference.py`)
**Goal:** Use the trained Diffusion model to generate the simulated contralateral breast (L->R and R->L).
**Inputs:** Diffusion checkpoint, Autoencoder.
**Output:** `AAD-Net/stage3_pseudo_labeling/inference.py`
**Action:**
*   Implement `run_bilateral_inference(diff_unet, autoencoder, L_cc, R_cc, L_mlo, R_mlo, steps=20)`.
*   **Pathway A:** Encode `L_cc` (condition). Start with full Gaussian noise. Run the DDIM scheduler for 20 steps to denoise using the U-Net. Decode to get **`hat_I_R`**.
*   **Pathway B:** Encode `R_cc` (condition). Run DDIM to denoise. Decode to get **`hat_I_L`**.
*   Return `hat_I_R, hat_I_L`.
**Verification:** Run with dummy tensors. Check output shapes match `(1, 1, 512, 1024)`.

---

#### Step 13: Implement Domain-Adapted Med-LPIPS (`stage3_pseudo_labeling/perceptual_loss.py`)
**Goal:** Calculate the structural/textural dissimilarity between the real image and the reconstructed image.
**Inputs:** Pre-trained VGG-16 (RadImageNet weights).
**Output:** `AAD-Net/stage3_pseudo_labeling/perceptual_loss.py`
**Action:**
*   Load VGG16 that was pre-trained on a medical grayscale dataset (RadImageNet).
*   Implement `compute_med_lpips_maps(L_cc, R_cc, hat_I_R, hat_I_L, patch_size=128)`.
*   **Critical Logic:** Extract multi-scale features from Blocks 1-5 of VGG.
*   For each patch, compute Cosine Distance between the features of `hat_I_R` and `R_cc`.
*   Weight shallow layers (Blocks 1,2) by 0.7, deep layers (Blocks 4,5) by 0.3 to prioritize texture distortion over macro-shape.
*   Return a `raw_asymmetry_map` (grid of patch-wise scores).
**Verification:** Manually compute LPIPS between an identical image and a different image. The different image must give a significantly higher score.

---

#### Step 14: Implement GMM Soft Labeling (`stage3_pseudo_labeling/statistical_gmm.py`)
**Goal:** Dynamically threshold patch scores to create Soft Pseudo-labels for the Weakly Supervised classifier.
**Inputs:** `raw_asymmetry_map` from Step 13.
**Output:** `AAD-Net/stage3_pseudo_labeling/statistical_gmm.py`
**Action:**
*   Implement `calculate_gmm_pseudo_labels(raw_asymmetry_maps, components=2)`.
*   **Logic:**
    1.  Load population mean and std from a reference file (calculated from 100 Normal patients) to convert raw scores to `Z-scores`.
    2.  Fit an unsupervised 2-component Gaussian Mixture Model (GMM) to the flattened Z-scores.
    3.  Extract the intersection of the two Gaussians. Set this as `T_GMM`.
    4.  Apply a scaled sigmoid: `P_k = Sigmoid((Z_k - T_GMM) / 0.1)` to generate soft labels `[0, 1]`.
    5.  Return `soft_pseudo_labels` dictionary containing `cc_map` and `mlo_map`.
**Verification:** Run on dummy scores containing two clearly separated distributions. Check that the output probabilities cleanly separate into near 0 and near 1.

---

#### Step 15: Implement Stage 3 Runner (`stage3_pseudo_labeling/run_stage3.py`)
**Goal:** Generate and save the soft pseudo-labels for all patients in the dataset.
**Inputs:** Steps 10, 12, 13, 14.
**Output:** `.npy` file at `processed/stage3_pseudolabels/pseudo_labels.npy`.
**Action:**
*   Load Stage 1 manifest and Stage 2 checkpoint.
*   Load `autoencoder` and `diff_unet`. Set `diff_unet.eval()`.
*   Loop through all patients in `manifest_stage1.json`.
*   For each patient:
    *   Run `run_bilateral_inference` to get `hat_I_R`, `hat_I_L`.
    *   Run `compute_med_lpips_maps` to get raw asymmetry scores.
    *   Run `calculate_gmm_pseudo_labels` to get probabilities.
    *   Store in a master dictionary `all_soft_labels[patient_id] = {...}`.
*   Save `all_soft_labels` as a `.npy` file.
**Verification:** Verify the `.npy` file is created and has the same number of keys as the manifest.

---

#### Step 16: Implement Stage 3 Verifier (`stage3_pseudo_labeling/verify_stage3.py`)
**Goal:** Ensure pseudo-labels have dynamic thresholds and structure.
**Action:**
*   Load the saved `.npy` file.
*   Loop through a few entries.
*   Check that `labels['cc_map'].max() > labels['cc_map'].min()` and `labels['cc_map'].max() <= 1.0`.
*   Print `✓ Stage 3 Pseudo-labels generated successfully.`

---

#### Step 17: Implement Supervised Contrastive Loss (`stage4_classifier/patch_supcon.py`)
**Goal:** Define the loss function to train the patch extractor backbone.
**Inputs:** None.
**Output:** `AAD-Net/stage4_classifier/patch_supcon.py`
**Action:**
*   Implement `SupConLoss(temperature=0.07)`.
*   Must accept `features` (batch, num_patches, dim) and `labels` (batch, num_patches) - where `labels` are the soft pseudo-labels.
*   Standard supervised contrastive loss: Pull patches with the same pseudo-label together, push patches with different pseudo-labels apart in the embedding space.
**Verification:** Pass dummy `features` and `labels` into the function. Ensure it returns a scalar loss without crashing.

---

#### Step 18: Implement Adaptive Multi-View Classifier (`stage4_classifier/fusion_network.py`)
**Goal:** Extract patch features, weigh them, fuse MLO and CC views, and classify.
**Inputs:** None (will use pre-trained backbone).
**Output:** `AAD-Net/stage4_classifier/fusion_network.py`
**Action:**
*   Implement `AdaptiveMultiViewClassifier(dropout)`.
*   `extract_patch_features(x_cc, x_mlo)`: Split image into 128x128 patches. Pass each patch through a ResNet-50 backbone (ensure patches are upscaled to 224x224 before entering ResNet). Return patch vectors.
*   `forward(x_cc, x_mlo, scores_cc, scores_mlo)`:
    1.  Extract patch features (using the backbone).
    2.  Compute view weights: `w_cc = max(scores_cc)`, `w_mlo = max(scores_mlo)`. Apply Softmax to get `alpha_cc` and `alpha_mlo`.
    3.  Aggregate patch features: `V_cc = mean(patch_vectors_cc)`, `V_mlo = mean(patch_vectors_mlo)`.
    4.  Fuse: `V_final = alpha_cc * V_cc + alpha_mlo * V_mlo`.
    5.  Pass `V_final` through 2 linear layers + Dropout + Softmax to get `logits`.
*   Return `logits`.
**Verification:** Run a dummy forward pass. Ensure output shape is `(batch_size, 2)`.

---

#### Step 19: Implement Stage 4 Runner (`stage4_classifier/run_stage4.py`)
**Goal:** Train the classifier using the Teacher-Student framework and Stage 3 pseudo-labels.
**Inputs:** Steps 17, 18, `manifest_stage1.json`, `pseudo_labels.npy`.
**Output:** `AAD-Net/checkpoints/classifier_student.pth`.
**Action:**
*   Instantiate `student = AdaptiveMultiViewClassifier()` and `teacher = deepcopy(student)`. Freeze `teacher` weights.
*   Initialize `optimizer`, `ce_loss`, `supcon_loss`.
*   Loop for `CLS_EPOCHS`:
    *   Iterate through `manifest_stage1.json`.
    *   Load `.pt` file and corresponding `.npy` pseudo-labels.
    *   Calculate `loss_supcon` (Step 17) using patch features.
    *   Get `logits_student` and `logits_teacher`. Calculate `loss_ce` with true labels and `loss_consistency` (KLDiv between student and teacher logits).
    *   `total_loss = 0.5*loss_supcon + 0.5*loss_ce + 0.05*loss_consistency`.
    *   Perform Gradient Accumulation (`loss / ACCUMULATION_STEPS`). Backward, clip gradients.
    *   Upon `ACCUMULATION_STEPS` reached: `optimizer.step()`, `optimizer.zero_grad()`.
    *   **Update Teacher:** Use Exponential Moving Average (EMA): `teacher_weights = 0.999 * teacher_weights + 0.001 * student_weights`.
*   Save `student.state_dict()` to checkpoint.
**Verification:** Checkpoint file is created. Loss values printed during epochs are decreasing.

---

#### Step 20: Implement Evaluator and Stage 4 Verifier (`stage4_classifier/evaluator.py` & `verify_stage4.py`)
**Goal:** Compute MedIA-standard metrics.
**Output:** `AAD-Net/stage4_classifier/evaluator.py` and `verify_stage4.py`.
**Action:**
*   Implement `compute_AUC(y_pred, y_true)`, `compute_Sensitivity_At_Specificity(y_pred, y_true, target_spec)`.
*   `verify_stage4.py`: Load the validation loader (which strictly excludes Bi-Rads 3).
*   Load `classifier_student.pth`, set `eval()`.
*   Run validation loop. Print `AUC` and `Sensitivity@95%`.
*   Assert `AUC > 0.5` (better than random, ensuring the model actually learned).
*   If passed, print `✓ Stage 4 Model Trained and Validated.`

---

#### Step 21: Implement the Orchestrator (`runner/run_all.py`)
**Goal:** Tie all stages together sequentially.
**Action:**
*   Write a script that uses `subprocess.run(["python", path_to_file])` to sequentially execute:
    1. `run_stage1.py`
    2. `verify_stage1.py`
    3. `run_stage2.py`
    4. `verify_stage2.py`
    5. `run_stage3.py`
    6. `verify_stage3.py`
    7. `run_stage4.py`
    8. `verify_stage4.py`
*   **Crucial check:** If any subprocess fails (return code != 0), immediately halt the orchestration and print the error message.
**Verification:** Run `python run_all.py`. After 50-100 minutes (depending on your GPU), it should output `🎉 Full AAD-Net Pipeline executed successfully.` and the final Clinical Metrics.

---

### 💡 A Note to the Agent on Implementation Strategy
1. **Start with Step 1 and Step 2.** Verify that your Data Loader correctly discards Bi-Rads 3. This is the single most important rule for your Weakly Supervised paper. Once this works, the rest of the pipeline becomes deterministic.
2. **Write "Dummy" Placeholders first:** When coding Steps 7, 8, and 17, if you don't have the specific pre-trained weights (e.g., VoxelMorph or VQ-GAN) on your machine, write a class that returns dummy tensors of the correct shape so you can debug Stage 2's architecture and GPU memory usage immediately. You can replace the dummy with the actual weights once you download them. 
3. **Always run the verification script** immediately after writing a `run_stage` script. This catches shape mismatches (e.g., `2048` vs `1024`) or label mapping errors before moving to the next costly phase.