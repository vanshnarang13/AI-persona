# Project: SAR to EO Image Translation

## Overview

Built a Hybrid Conditional GAN that translates Sentinel-1 SAR (VV polarization) radar patches into matching Sentinel-2 optical (RGB) imagery. The task is fundamentally ill-posed: SAR carries structural information (edges, boundaries, field lines) but essentially no color information — measured correlation between SAR brightness and optical brightness was about -0.04, i.e. near zero. So the model has to recover geometry directly from the SAR while learning colour/tone from priors. The model is ranked on perceptual metrics (LPIPS, FID) with pixel metrics (SSIM, PSNR) reported as secondary diagnostics, and generalization is tested on a held-out set of unseen geographies.

## Architecture

A dual-branch generator (~14M params) fuses a CNN branch (nine Res2Net-style residual blocks with squeeze-and-excitation, for local texture) with a Transformer branch (twelve blocks, for global context), connected bidirectionally: a downsampling transmission feeds encoder features into the Transformer as tokens, and an upsampling transmission projects Transformer features back into the CNN's residual blocks. A class token drives an auxiliary terrain-classification head that injects a colour prior during training. Discrimination is done by a multi-scale spectral-norm PatchGAN operating at three resolutions. Trained on 21,339 scene-disjoint SAR/optical patch pairs across seven terrains (combining Kaggle's terrain-segregated Sentinel-1/2 set with a SEN12MS subset), with adversarial + L1 + VGG-perceptual + terrain-classification losses. I ran the baseline (global ViT attention) against two controlled ablations — swapping in windowed Swin attention, and halving the L1 weight — trained on Modal cloud GPUs (H100), tracked in Weights & Biases.

## Results

On the full 3,321-image test split, all three variants (LPIPS 0.430-0.434, FID 98-105) clearly beat published Pix2Pix baselines on this kind of data (~LPIPS 0.483). Halving the L1 weight gave the best LPIPS on both validation and test without hurting pixel metrics, suggesting the standard L1 weight was mildly over-smoothing perceptual detail. Qualitatively, high-frequency structure (edges, field boundaries, river networks) transfers correctly since that information genuinely exists in the SAR; colour and tone are where outputs drift, since that's the part SAR doesn't determine and the model has to infer.

## Contribution

Solo project. I designed the experiment structure (one held-fixed baseline, two clean ablations on the same data/settings), built the full training/eval/inference pipeline, wrote the scene-disjoint dataset split logic to prevent leakage from overlapping patch crops, and made the deliberate call to train on VGG perceptual loss rather than LPIPS specifically because LPIPS is a ranked evaluation metric — training on it would be metric gaming.

## Links
GitHub: https://github.com/vanshnarang13/sar-to-eo
