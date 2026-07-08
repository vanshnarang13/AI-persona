# GitHub Commit History: SAR to EO Image Translation

Repository: https://github.com/vanshnarang13/sar-to-eo
Last 7 commits (most recent first)

## Commit b59a8949 — 08 Jul 2026
Author: Vansh Narang

Message: Exclude notebooks from GitHub language stats

Files changed (1):
  added      .gitattributes  (+5 -0)

## Commit 6a5fa1e1 — 05 Jul 2026
Author: vanshnarang13

Message: README updated

Files changed (1):
  modified   README.md  (+1 -1)

## Commit bda6a982 — 05 Jul 2026
Author: vanshnarang13

Message: README updated

Removed the internship title from the README.

Files changed (1):
  modified   README.md  (+0 -2)

## Commit 14c4c7fb — 05 Jul 2026
Author: vanshnarang13

Message: readme updated

Files changed (7):
  modified   README.md  (+184 -63)
  added      assets/discriminator_multiscale.png  (+0 -0)
  added      assets/discriminator_patchgan.png  (+0 -0)
  added      assets/ds_transmission.png  (+0 -0)
  added      assets/generator.png  (+0 -0)
  added      assets/residual_block.png  (+0 -0)
  added      assets/us_transmission.png  (+0 -0)

## Commit 4ef0058b — 01 Jul 2026
Author: vanshnarang13

Message: Clean up .gitignore 

Removed report artifacts and internal notes from .gitignore.

Files changed (1):
  modified   .gitignore  (+0 -13)

## Commit e3a8c1d4 — 27 Jun 2026
Author: vanshnarang13

Message: Add model weights download link to README

Files changed (3):
  modified   README.md  (+4 -7)
  modified   src/models/hybrid_cgan.py  (+1 -6)
  modified   src/models/swin_blocks.py  (+1 -17)

## Commit cf567071 — 27 Jun 2026
Author: vanshnarang13

Message: Initial commit: SAR-to-EO image translation (Hybrid cGAN)

Hybrid CNN-Transformer conditional GAN translating Sentinel-1 SAR (VV) to
Sentinel-2 optical (RGB). Includes the training/eval/inference pipeline,
three ablation configs (ViT / Swin / Swin L1=40), Modal cloud entrypoints,
a flat sample test set for reproducing metrics, and the EDA notebook.
Datasets, model checkpoints, source papers, and W&B logs are gitignored;
checkpoints are hosted separately.

Files changed (300):
  added      .gitignore  (+46 -0)
  added      README.md  (+179 -0)
  added      configs/hybrid_cgan_swin.yaml  (+67 -0)
  added      configs/hybrid_cgan_swin_l1_40.yaml  (+67 -0)
  added      configs/hybrid_cgan_vit.yaml  (+63 -0)
  added      eval.py  (+65 -0)
  added      infer.py  (+91 -0)
  added      modal_app.py  (+464 -0)
  ... and 292 more files
