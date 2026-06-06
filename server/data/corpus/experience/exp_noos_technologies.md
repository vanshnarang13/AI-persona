# Experience: AI Research Intern — Noos Technologies

## Role and Duration

AI Research Intern at Noos Technologies from June 2025 to August 2025, a three month engagement focused entirely on deep learning research in digital image steganography.

## What the Work Was

Image steganography is the science of hiding a secret binary payload inside a cover image in a way that is visually imperceptible to a human observer but recoverable by a trained decoder network. The goal is that an attacker looking at the image cannot tell it carries hidden data, but someone with the right decoder can extract the exact payload bits with high accuracy.

This is a genuinely hard problem because you are optimizing for three competing objectives simultaneously. You want high payload capacity (hiding as many bits as possible), high robustness (the hidden data survives JPEG compression, cropping, brightness changes, and other distortions the image might encounter), and high visual fidelity (the modified image looks identical to the original). These three goals trade off against each other in fundamental ways. Increasing payload capacity tends to introduce visible artifacts. Making the encoding more robust usually requires embedding stronger signals, which also makes them more visible.

## What I Actually Built

### HiDDeN

HiDDeN (Hiding Data with Deep Networks) is an encoder-decoder architecture where the encoder takes a cover image and a message bitstring and outputs a stego image, and the decoder takes the stego image and outputs the recovered bitstring. The training setup includes a noise layer between the encoder and decoder that applies differentiable approximations of real-world distortions (JPEG compression, dropout, Gaussian blur). This forces the encoder to embed signals that survive those distortions.

I implemented the full training pipeline from scratch in PyTorch. The loss function combines a reconstruction term (the stego image should look like the cover image, measured by MSE and SSIM), a message accuracy term (the decoder should recover the exact bits), and optionally an adversarial term to further suppress detectability. Getting the balance between these loss components right was the main challenge. Too much weight on reconstruction and the payload becomes fragile. Too much weight on accuracy and the encoder starts producing visible artifacts.

### SteganoGAN

SteganoGAN replaces the noise layer approach with a GAN training setup. A generator embeds the payload, a discriminator tries to distinguish stego images from clean images, and the encoder is trained adversarially to fool the discriminator while still recovering the payload accurately. The discriminator provides a richer training signal than a simple pixel loss because it learns what makes stego images look artificial, not just what the pixel difference is.

The main challenge with SteganoGAN training is the instability typical of GANs. Mode collapse, discriminator domination, and oscillating losses are all real problems. I worked through gradient penalty regularization, careful learning rate scheduling, and monitoring per-component losses separately to keep training stable.

### CAISFormer

CAISFormer is a transformer-based steganography model. Where HiDDeN and SteganoGAN use convolutional backbones, CAISFormer uses attention mechanisms to decide where in the image to embed payload bits based on global image context. The intuition is that a transformer can learn which image regions are visually complex enough to absorb a perturbation without it being noticeable, and which regions (like smooth backgrounds) are fragile.

Transformer models show better compression resilience than CNN-based approaches in this domain. Compression artifacts are spatially structured, and attention mechanisms handle spatial dependencies more naturally than convolutional sliding windows.

## What I Measured

For each model I computed PSNR (Peak Signal to Noise Ratio, measures image quality, higher is better, typically aiming for above 35 dB), SSIM (Structural Similarity Index, perceptual quality metric, ranges 0 to 1), and bit accuracy (the percentage of payload bits correctly recovered by the decoder after distortions).

I ran systematic trade-off analysis across these metrics as I varied payload size and noise layer configuration. The key finding was that the capacity-robustness-fidelity triangle is real and fundamental, not an artifact of any particular architecture. Every model showed the same general pattern.

## Tech Stack

PyTorch, encoder-decoder CNN architectures, GANs, transformer architectures, custom loss functions (reconstruction loss, adversarial loss, perceptual loss), PSNR and SSIM evaluation, Python.

## What I Learned and What I Would Do Differently

The main skill this internship built was reading ML research papers and implementing them faithfully. That means understanding not just the architecture diagram but also the training setup, the specific hyperparameters, the evaluation protocol, and the subtle tricks that make the model actually converge. None of that is fully in the paper. You have to figure it out.

If I were doing this project again, I would build a unified evaluation harness from the beginning instead of maintaining separate evaluation scripts for each model.

Initially, each architecture (HiDDeN, SteganoGAN, and CAISFormer) had slightly different evaluation pipelines and preprocessing assumptions. That made fair comparison difficult because improvements could come from evaluation differences rather than the model itself.

I would standardize:

identical datasets and train/validation splits
identical distortion pipelines (JPEG, crop, blur, resize)
common metrics (PSNR, SSIM, bit recovery accuracy, payload capacity)
automated experiment tracking and comparison dashboards

That would make ablations and benchmarking significantly faster and more reproducible.

I would also explore adaptive payload allocation instead of fixed payload sizing.

In the current setup, every image carries the same amount of hidden information regardless of visual complexity. But images are not uniform—textured and high-frequency regions can typically hide more information than smooth regions without introducing visible artifacts.

A more advanced approach would dynamically allocate payload capacity based on local image complexity or learned attention maps, potentially improving both visual fidelity and robustness at the same time.

## Why It Matters for AI Engineering Roles

This internship demonstrates the ability to go from a research paper to a working implementation without someone handing you the code. Debugging convergence failures requires understanding what is happening mathematically, not just running more experiments. That combination of research depth and practical implementation skill is directly relevant to any role that involves working with ML systems at the model level.
