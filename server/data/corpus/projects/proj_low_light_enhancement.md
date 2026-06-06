# Project: Low-Light Image Denoiser

## Overview
Built a deep learning model for low-light image enhancement and denoising, based on the ImageLab-style architecture used for the NTIRE 2024 Low Light Image Enhancement setting. The model takes a dark/noisy RGB image as input and predicts an enhanced image that is brighter, cleaner, and structurally closer to the paired high-light ground truth.

The implementation is notebook-driven and uses a custom PyTorch architecture combining three ideas:
- Spatial enhancement through CoordConv + SCPA blocks
- Multi-scale reconstruction through a U-Net-like autoencoder with residual dense attention blocks
- Noise suppression through a parallel denoising branch with inverted residual blocks and attention

The notebook reports an average validation PSNR of **24.58 dB** after loading pretrained weights and fine-tuning/evaluating on the paired validation set.

## Tech Stack
Python, PyTorch, torchvision, torchmetrics, PIL/Pillow, OpenCV, NumPy, pandas, matplotlib, tqdm, Kaggle Notebook, NVIDIA Tesla P100

## Repository Structure

```
Low-Light-Image-Denoiser/
├── LowLightImageDeNoiser.ipynb       # Full model, training, and evaluation notebook
├── model_weights.pth                 # Saved PyTorch model weights
├── Low-Light Image Denoising.docx    # Architecture/training report
└── README.md
```

## Dataset Layout

The notebook expects paired low-light and high-light images with matching filenames:

```
/kaggle/input/dataset00/augmented_Train/
├── augmented/
│   ├── low/
│   └── high/
└── val/
    ├── low/
    └── high/
```

Images are loaded with PIL, converted to RGB, resized to `400 x 592`, and converted to tensors. The dataset class returns `(low_image, high_image)` pairs for supervised image-to-image training.

## Architecture

```
Low-light RGB image
      │
      ├─ Denoise Branch
      │    ├─ ConvBlock
      │    ├─ 4 inverted residual blocks
      │    ├─ channel/spatial attention gate
      │    └─ ConvBlock
      │
      ├─ SCPA Branch
      │    ├─ CoordConv: RGB + x/y coordinate channels
      │    ├─ 5 SCPA blocks
      │    └─ 3-channel convolution
      │
      └─ Autoencoder Branch
           ├─ input = original image + SCPA output
           ├─ 4 encoder stages with residual dense attention
           ├─ skip connections
           └─ 4 decoder stages with transposed convolution upsampling

Final output = Conv(autoencoder output) + denoise branch output
```

## SCPA Branch

The SCPA branch starts with a CoordConv layer. CoordConv appends normalized x/y coordinate maps to the image tensor before convolution, letting the network learn spatially aware enhancement patterns instead of relying only on local RGB neighborhoods.

After CoordConv, the branch applies five SCPA blocks. Each SCPA block has two paths:
- A standard convolutional path with `1x1` then `3x3` convolution
- A pixel-attention path that uses a sigmoid gate to modulate convolutional features

The two branches are summed, projected through a final `1x1` convolution, and added back to the input as a residual output. This helps preserve structure while enhancing low-light details.

## Denoising Branch

The denoising branch runs in parallel with the enhancement branch. It uses:
- An initial Conv + BatchNorm + ReLU block
- Four inverted residual blocks with expansion factor 6
- An attention block that learns a single-channel attention mask
- A final Conv + BatchNorm + ReLU block

This branch produces a denoised RGB-like residual that is added to the enhanced output at the end. The goal is to suppress low-light noise without washing out edges and texture.

## Residual Dense Attention Autoencoder

The autoencoder is a modified U-Net. It uses four encoder stages and four decoder stages:

```
Encoder: 3 → 32 → 64 → 128 → 256
Decoder: 256 → 128 → 64 → 32 → 3
```

Each encoder/decoder stage contains two Residual Dense Attention (RDA) blocks. An RDA block combines:
- A residual block with two `3x3` convolutions
- A dense block with four growth layers
- A spatial attention block over average-pooled and max-pooled channel features

The decoder uses transposed convolutions for upsampling and concatenates skip features from matching encoder levels. This lets the model recover fine spatial detail while still using deeper context from the bottleneck.

## Loss Function

The notebook implements a combined image reconstruction loss:

```
L = L1 + gradient_loss + 0.1 * SSIM_loss
```

The components are:
- **L1 loss**: keeps predicted pixel values close to the high-light target
- **SSIM loss**: encourages structural similarity in luminance, contrast, and texture
- **Gradient loss**: compares x/y image gradients so edges and texture remain sharp

The gradient loss is computed with L1 distance between adjacent-pixel differences in the prediction and target along both horizontal and vertical axes.

## Training Setup

The notebook uses:
- Adam optimizer
- Learning rate `1e-4` in the fine-tuning cell
- Batch size `2`
- Paired train and validation loaders
- Saved checkpoint loading from `model_weights.pth`
- Periodic checkpointing every two epochs during the shown fine-tuning loop

The accompanying report describes the broader training setup as Adam optimization on Kaggle with an NVIDIA Tesla P100 GPU, with learning rate decay over a longer 100-epoch training run. The notebook itself shows a loaded-weight fine-tuning run for 5 epochs.

## Evaluation

The notebook defines PSNR directly from MSE:

```
PSNR = 20 * log10(1.0 / sqrt(MSE))
```

It evaluates the model over the validation low/high image pairs and reports:

```
Average PSNR: 24.58 dB
```

The notebook's shown fine-tuning logs also report train PSNR values around 24 dB:

| Epoch | Train Loss | Train PSNR |
|---|---:|---:|
| 1 | 0.1361 | 24.16 dB |
| 2 | 0.1316 | 24.55 dB |
| 3 | 0.1321 | 24.47 dB |
| 4 | 0.1295 | 24.72 dB |
| 5 | 0.1311 | 24.64 dB |

## Key Techniques

**CoordConv for spatial awareness**: Low-light artifacts can vary across image regions. Adding coordinate channels gives the model explicit location information.

**SCPA-style pixel attention**: The SCPA branch lets the network emphasize important local regions while keeping a residual path for stable enhancement.

**Parallel denoising residual**: Noise suppression is handled by a dedicated branch instead of forcing the autoencoder alone to solve enhancement and denoising simultaneously.

**U-Net skip connections**: Encoder features are concatenated into decoder stages, preserving fine detail that would otherwise be lost through downsampling.

**Residual dense attention blocks**: The model combines residual learning, dense feature reuse, and spatial attention inside each autoencoder stage.

**Perceptual reconstruction objective**: L1, SSIM, and gradient losses optimize complementary qualities: pixel accuracy, structure, and edge sharpness.

## Results & Findings

- The saved-weight notebook evaluation reports **24.58 dB average PSNR** on the validation image pairs.
- The fine-tuning logs show train PSNR stabilizing around **24-25 dB** over the displayed 5-epoch run.
- The architecture is designed to avoid the common low-light enhancement failure mode where brightness improves but texture becomes blurry. The gradient loss and skip-connected autoencoder directly target this.
- The three-branch design separates enhancement, multi-scale reconstruction, and denoising, which makes the model more expressive than a plain U-Net baseline.

## Challenges & Solutions

**Low-light noise vs detail preservation**: Brightening dark images can amplify sensor noise. The solution was a separate denoising branch with inverted residual blocks and attention.

**Spatially uneven illumination**: Different image regions may need different enhancement strength. CoordConv and spatial/pixel attention help the model adapt enhancement by location and content.

**Blurry outputs from pixel-only losses**: L1 alone can produce smooth predictions. Adding SSIM and gradient loss encourages structural similarity and sharper edges.

**Recovering fine details after downsampling**: The autoencoder uses U-Net-style skip connections and residual dense attention blocks to preserve fine-scale features.

**Notebook-based experimentation**: The project was developed in a Kaggle notebook, so the model, training loop, metrics, and visual inspection code live in one file. That is practical for competition experimentation but harder to package for reuse.

## What I'd Do Differently

1. Refactor the notebook into separate `dataset.py`, `model.py`, `losses.py`, `train.py`, and `eval.py` modules for reproducibility.
2. Add a validation checkpointing loop that saves the best model by validation PSNR/SSIM instead of only periodic epoch checkpoints.
3. Track SSIM alongside PSNR during evaluation, since PSNR alone does not fully capture perceived enhancement quality.
4. Add ablations for the SCPA branch, denoising branch, CoordConv, and gradient loss to quantify each component's contribution.
5. Add mixed-precision training and patch-based training for larger images, since full-resolution low-light enhancement is memory-intensive.
6. Export an inference script that takes a folder of low-light images and writes enhanced outputs without requiring notebook execution.

## Links
GitHub: https://github.com/akshitmanocha/Low-Light-Image-Denoiser
Reference paper: https://arxiv.org/pdf/2404.14248
