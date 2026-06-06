# GitHub README: Drone Audio Classification

Repository: https://github.com/vanshnarang13/Drone-Audio-Classification

# Drone Audio Classification

> Multi-task deep learning system that classifies drone audio recordings across three simultaneous targets — drone model identity, maneuvering direction, and mechanical fault type — using MFCC features fed into a shared 1D CNN with three output heads.

---

## Overview

Identifying a drone's operational state from raw audio has applications in airspace security, autonomous UAV maintenance diagnostics, and search-and-rescue coordination. This project frames drone audio analysis as a **multi-output classification** problem: given a single `.wav` recording from a microphone, simultaneously predict (1) which drone model is flying, (2) the direction it is maneuvering, and (3) whether a mechanical fault is present and of what type.

The pipeline extracts Mel-Frequency Cepstral Coefficients (MFCCs) from each audio file, uses a shared 1D CNN backbone to learn spectral-temporal patterns, then branches into three independent softmax output heads — one per classification task.

---

## Dataset

| Property | Value |
|---|---|
| Audio format | `.wav` files, multi-microphone capture (mic1, mic2) |
| Splits | Training, Validation, Test (mic1 + mic2 test sets) |
| Test set size | 64,774 audio clips |
| Label encoding | Parsed directly from structured filename convention |

**Filename convention:** `<DroneModel>_<Direction>_<Fault>_mic<N>_<idx>_<background>_<id>_snr=<value>.wav`

**Class labels:**

| Task | Classes |
|---|---|
| Drone Model | A, B, C — 3 classes |
| Maneuvering Direction | Right (R), Left (L), Forward (F), Backward (B), Clockwise (CC), Counter-clockwise (C) — 6 classes |
| Mechanical Fault | PC1, PC2, PC3, PC4, MF1, MF2, MF3, MF4, Normal (N) — 9 classes |

---

## Approach / Methodology

### 1. Label Extraction
Labels are parsed from the structured filename using string splitting. `extract_labels()` splits on underscores and maps string tokens to integer class indices via dictionaries for all three tasks.

### 2. MFCC Feature Extraction
Each audio file is loaded with `librosa.load()` and processed with `librosa.feature.mfcc()`. The time axis is mean-pooled (`np.mean(..., axis=0)`) to produce a fixed-length feature vector regardless of clip duration. The number of MFCC coefficients was swept across experiments: 40, 50, 70, and 100.

### 3. Multi-Output Neural Network Architecture
A shared **1D CNN** backbone processes the MFCC feature vector:

```
Input (n_mfcc,) → Reshape (n_mfcc, 1)
→ Conv1D(32, kernel=3, stride=2) + BatchNorm + MaxPool
→ Conv1D(64, kernel=3)           + BatchNorm + MaxPool
→ Conv1D(128, kernel=3)          + BatchNorm + MaxPool
→ Flatten → Dense(256) + BatchNorm + Dropout
→ [Head 1] Dense(3,  softmax)   — drone_model
→ [Head 2] Dense(6,  softmax)   — direction
→ [Head 3] Dense(9,  softmax)   — fault
```

All three heads share the convolutional feature extractor. The model is compiled with separate categorical cross-entropy losses per head using TensorFlow's functional API.

### 4. Experiment Sweep
Nine model variants were trained, varying:
- MFCC count: 40 / 50 / 70 / 100
- Activation: ReLU vs LeakyReLU
- Dropout rate: 0.5 vs 0.25
- Optimizer: RMSprop vs Adam
- Training epochs: 10 / 30 / 50

Predictions from each experiment were saved as versioned CSV files.

### 5. Inference
Final predictions on the test set (64,774 clips) are produced by `model.predict()` and decoded back to label strings using `np.argmax` over each output head.

---

## Tech Stack

| Category | Libraries / Tools |
|---|---|
| Audio processing | `librosa` |
| Deep learning | `TensorFlow`, `tf.keras` (functional API) |

---

## Results

Best performing model configuration (Model 9 — 100 MFCC features, LeakyReLU, Adam optimizer, 30 epochs):

| Task | Train Accuracy | Validation Accuracy |
|---|---|---|
| Drone Model | 99.81% | 99.93% |
| Maneuvering Direction | 89.69% | 90.57% |
| Mechanical Fault | 93.62% | 94.33% |

**Progressive improvement across experiments:**

| Model | MFCC Features | Val Direction Acc | Val Fault Acc |
|---|---|---|---|
| 3 (baseline) | 40 | 83.0% | 89.8% |
| 5 | 50 | 83.9% | 90.3% |
| 6 | 70 (ReLU) | 87.6% | 92.1% |
| 7 (LeakyReLU) | 70 | 86.6% | 92.4% |
| 8 (Adam, lower dropout) | 70 | 89.9% | 93.9% |
| 9 (final) | 100 | 90.6% | 94.3% |

Drone model classification consistently achieved >99.8% accuracy across all experiments. Direction and fault classification showed the largest gains from increasing MFCC resolution and switching to the Adam optimizer.

---

