# Project: Drone Audio Classification

## Overview

This project builds a multi-task deep learning system to classify drone audio recordings across three simultaneous prediction targets: drone model identity (three classes: A, B, or C), maneuvering direction (six classes: right, left, forward, backward, clockwise, counter-clockwise), and mechanical fault type (nine classes: four propeller crack variants, four motor fault variants, and normal).

The system takes raw audio .wav files as input, extracts MFCC (Mel-Frequency Cepstral Coefficients) features, and feeds them through a shared 1D CNN backbone with three parallel softmax output heads, one per task. The model is trained end-to-end on multi-output categorical cross-entropy loss.

GitHub: https://github.com/vanshnarang13/Drone-Audio-Classification

## Tech Stack

Python, TensorFlow, Keras functional API, Conv1D (1D convolutional neural network), BatchNormalization, MaxPooling1D, Dense layers, Dropout, LeakyReLU and ReLU activations, softmax outputs, RMSprop and Adam optimizers, categorical cross-entropy loss, librosa (audio feature extraction), NumPy, Pandas, Matplotlib.

## Dataset

Audio format: .wav files with dual microphone capture (mic1 and mic2 recorded separately). The dataset provides training, validation, and test splits, with the test set containing 64,774 audio clips.

Labels are encoded in the filename using a structured naming convention: DroneModel_Direction_Fault_micN_idx_background_id_snr=value.wav. This means every label can be parsed by splitting the filename string on underscores, without needing a separate label file.

Drone models: A, B, C (three classes). Maneuvering directions: R (right), L (left), F (forward), B (backward), CC (clockwise), C (counter-clockwise) (six classes). Mechanical faults: PC1, PC2, PC3, PC4 (propeller crack variants), MF1, MF2, MF3, MF4 (motor fault variants), N (normal) (nine classes). Background environments and SNR values are also encoded in filenames, providing real-world noise condition context.

## Architecture and Approach

### Label Extraction

Because all labels are encoded in filenames, the first step is filename parsing. String splitting on underscores, combined with three separate dictionaries mapping token strings to integer class indices, handles all three tasks cleanly. This was simpler and more reliable than a label CSV approach.

### MFCC Feature Extraction

Each audio file is loaded with librosa.load() and MFCC features are extracted using librosa.feature.mfcc(). MFCCs represent the spectral envelope of the audio signal, essentially capturing the timbre and texture of the sound. They are the standard feature representation for audio classification tasks because they correlate with perceptually meaningful acoustic properties.

The MFCC output is a matrix of shape (n_mfcc, time_steps) where time_steps varies with clip length. To produce a fixed-length input vector for the neural network, the time axis is mean-pooled using np.mean(axis=0), producing an n_mfcc-dimensional vector per clip. This discards temporal dynamics but produces a consistent input shape regardless of clip duration.

### Multi-Task CNN Architecture

The model uses a shared backbone followed by three independent output heads.

Shared backbone: three Conv1D layers with 32, 64, and 128 filters respectively, each followed by BatchNormalization and MaxPooling1D. The backbone is followed by a Dense(256) layer with BatchNormalization and Dropout. This shared representation learns features that are useful across all three classification tasks simultaneously.

Output heads: three parallel Dense layers with softmax activation, one for drone model (3 units), one for direction (6 units), and one for fault type (9 units). Each head has its own categorical cross-entropy loss, and Keras sums the losses across heads during training.

### Experiment Sweep

Nine model variants were tested by varying four hyperparameters: MFCC coefficient count (40, 50, 70, or 100), activation function (ReLU or LeakyReLU), dropout rate (0.5 or 0.25), and optimizer (RMSprop or Adam). Each variant was trained for 10, 30, or 50 epochs and evaluated on the validation set.

## Results

Best model was Model 9: 100 MFCC features, LeakyReLU activations, Adam optimizer, dropout 0.25, 30 epochs.

Drone model classification accuracy: 99.81% training, 99.93% validation. Direction classification accuracy: 89.69% training, 90.57% validation. Mechanical fault classification accuracy: 93.62% training, 94.33% validation.

Progressive improvement across the experiment sweep:

The baseline (Model 3, 40 MFCCs, RMSprop): 83.0% direction, 89.8% fault. Adding more MFCCs to 50 (Model 5): minimal change. Switching to 70 MFCCs and ReLU (Model 6): 87.6% direction, 92.1% fault. Switching to LeakyReLU at 70 MFCCs (Model 7): similar direction accuracy, slightly better fault. Switching from RMSprop to Adam at 70 MFCCs (Model 8): 89.9% direction, 93.9% fault. This was the biggest single improvement. Adding more MFCCs to 100 with Adam (Model 9): 90.6% direction, 94.3% fault.

The key findings were: drone model identity was trivially classifiable at above 99.8% throughout all experiments, which means the acoustic signature of drone A versus B versus C is extremely distinctive. Direction and fault classification drove all the meaningful iteration. The switch from RMSprop to Adam provided the largest accuracy gain (around two percentage points on direction). Increasing MFCC resolution from 40 to 100 provided a consistent but smaller gain.

## Design Tradeoffs

### Mean-Pooling the Time Axis

Averaging MFCC features over the time axis to get a fixed-length vector is a significant simplification. It discards all temporal structure in the audio. A drone executing a right turn has a temporal signature where the rotor speed changes over time. Mean-pooling collapses that temporal pattern into a single number.

The tradeoff was made consciously because it kept the model simple and the accuracy was still high enough to demonstrate the approach. But this is why direction classification accuracy (90%) is lower than fault accuracy (94%). Fault classification correlates with steady-state acoustic properties, which mean-pooled MFCCs capture well. Direction classification correlates with temporal dynamics, which they discard.

### Multi-Task vs Single-Task Models

Training one model on all three tasks simultaneously is more parameter-efficient than three separate models. The shared backbone learns representations that transfer across tasks. The risk is that if the tasks have very different loss scales, the gradients from the easy task can dominate and slow learning for the harder tasks.

This was partially visible in training: drone model accuracy saturated almost immediately (it is a trivially easy task acoustically), while direction and fault accuracy improved throughout training. BatchNormalization helped stabilize the gradient flow across tasks.

## What I Would Do Differently

The most important change would be replacing mean-pooled MFCCs with a sequence model. An LSTM or a Transformer encoder operating on the full MFCC sequence would preserve the temporal dynamics that mean-pooling discards. Directional flight patterns have time-varying acoustic signatures and a sequence model would capture those directly. I would expect a meaningful improvement in direction classification accuracy specifically.

The second change would be explicit loss weighting or gradient balancing across the three output heads. The current setup sums losses equally across tasks, which means the easy task (drone model, trivially classifiable) produces gradients that are not very informative but still contribute to each weight update. Down-weighting the loss from saturated tasks and up-weighting the loss from harder tasks would focus the optimization where it matters.
