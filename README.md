# Speech Processing Algorithms for Speech / Non-Speech Detection

A Voice Activity Detection (VAD) project implementing and evaluating four speech/non-speech detection algorithms — from classical signal processing to deep learning — benchmarked on a custom keyword dataset and the Google AVA-Speech corpus.

## Authors

| Student ID | Name |
|------------|------|
| 522K0026 | Nguyen Duc Anh |
| 522K0045 | Keni Nicholas Ondang |
| 522K0050 | Thant Thiri Maung |

---

## Project Structure

```
speechprocessing_final_project/
│
├── dataset/
│   ├── train/                             # Custom keyword recordings
│   │   ├── KeniNicholasOndang/
│   │   ├── ThantThiriMaung/
│   │   └── NguyenDucAnh/
│   └── ava_speech_trim/                   # AVA Speech benchmark dataset
│       └── ava_speech_labels_v1.csv
│
├── notebooks/
│   └── 522K0026_522K0045_522K0050.ipynb   # Main analysis notebook
│
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Files excluded from version control
└── README.md                               # This file
```

---

## Datasets

### Custom Keyword Dataset (`dataset/train/`)

Recorded by the three project members to cover keywords with diverse phonetic characteristics — voiced vowels, unvoiced fricatives, and plosives.

- **3 speakers:** Keni Nicholas Ondang, Thant Thiri Maung, Nguyen Duc Anh
- **4 keywords:** `yes`, `yeah`, `present`, `here`
- **10 recordings per keyword per speaker** → 120 total `.wav` files
- **Audio format:** 16,000 Hz mono WAV, ~1.88 s average duration

File naming convention:
```
word_index_StudentName.wav
# e.g. yes_1_NguyenDucAnh.wav
```

### AVA Speech Dataset (`dataset/ava_speech_trim/`)

Used as training and evaluation data for the statistical (GMM) and deep learning (DNN) algorithms. Recorded in real-world, unconstrained environments.

- **Label file:** `ava_speech_labels_v1.csv`
- **Columns:** `id`, `start`, `end`, `label`
- **Classes:** `CLEAN_SPEECH`, `SPEECH_WITH_NOISE`, `SPEECH_WITH_MUSIC`, `NO_SPEECH`
- **Source:** [AVA Speech — Google Research](https://research.google.com/ava/download.html)

---

## Algorithms Implemented

### Algorithm 1 — Energy-Based Detection (STE)
Classifies frames purely on loudness via Short-Time Energy. A frame is labelled Speech if its RMS energy (in dB) exceeds a fixed threshold. Fast and interpretable, but struggles to separate loud noise from quiet speech.

- **Feature:** RMS energy
- **Threshold:** `STE_THRESHOLD = 8 dB`

### Algorithm 2 — Hybrid STE + ZCR Detection
Extends Algorithm 1 by adding Zero Crossing Rate as a second gate, targeting unvoiced consonants (e.g., /s/, /f/) that are low-energy but high-frequency. A hangover mechanism smooths brief gaps between speech bursts.

- **Features:** RMS energy + Zero Crossing Rate
- **Thresholds:** `HYBRID_ENERGY_THRESHOLD = 8 dB`, `ZCR_THRESHOLD = 0.2`
- **Hangover:** `HANGOVER = 5` frames

### Algorithm 3 — Statistical Classification (GMM)
Shifts from deterministic thresholds to probabilistic modelling. MFCCs are extracted from each frame and fed into a pair of Gaussian Mixture Models (one per class). A frame is classified by comparing the log-likelihood of each model.

- **Features:** 13 MFCCs + 26 Mel filterbanks
- **Model:** `N_GMM_COMPONENTS = 16` per class
- **Training data:** Google AVA-Speech

### Algorithm 4 — Deep Neural Network (DNN)
A supervised sequence model trained end-to-end on AVA-Speech features. Takes fixed-length sequences of acoustic frames and outputs a binary VAD decision, learning complex non-linear boundaries that handcrafted rules cannot capture.

- **Input:** Sequences of length `SEQUENCE_LENGTH = 125` frames
- **Training data:** Google AVA-Speech

---

## Evaluation Summary

All four algorithms were benchmarked using Accuracy, Precision, Recall, and F1-Score (micro-averaged) on the custom keyword dataset via a Speaker Recognition Module (SRM) pipeline.

| Algorithm | Key Strength | Key Weakness |
|-----------|-------------|--------------|
| 1 — STE | Highest accuracy (~90%), cleanest signal | Misses quiet/unvoiced speech |
| 2 — Hybrid STE+ZCR | Captures more phonetic content | Over-sensitive; admits noise |
| 3 — GMM | Probabilistic, generalises to unseen noise | Misclassifies silence as speech |
| 4 — DNN | Learns complex patterns end-to-end | Requires large labelled data |

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd speechprocessing_final_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the datasets

```bash
# Custom recordings
mkdir -p dataset/train/KeniNicholasOndang
mkdir -p dataset/train/ThantThiriMaung
mkdir -p dataset/train/NguyenDucAnh
# Copy your .wav files into the matching speaker folder

# AVA Speech
mkdir -p dataset/ava_speech_trim
# Copy ava_speech_labels_v1.csv and trimmed audio clips here
```

### 4. Run the notebook

```bash
jupyter notebook notebooks/522K0026_522K0045_522K0050.ipynb
```

---

## Key Configuration Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SAMPLE_RATE` | 16,000 Hz | Audio sampling rate |
| `FRAME_LENGTH` | 512 samples | Short-time analysis window |
| `HOP_LENGTH` | 128 samples | Frame step size (FRAME_LENGTH / 4) |
| `RANDOM_SEED` | 42 | Global reproducibility seed |
| `STE_THRESHOLD` | 8 dB | Energy threshold (Algorithm 1) |
| `ZCR_THRESHOLD` | 0.2 | Zero-crossing threshold (Algorithm 2) |
| `HANGOVER` | 5 frames | Smoothing window (Algorithm 2) |
| `HYBRID_ENERGY_THRESHOLD` | 8 dB | Energy gate for hybrid (Algorithm 2) |
| `N_MFCC` | 13 | MFCC coefficients (Algorithm 3) |
| `N_MELS` | 26 | Mel filterbanks (Algorithm 3) |
| `N_GMM_COMPONENTS` | 16 | GMM mixture components (Algorithm 3) |
| `GMM_MAX_ITER` | 50 | GMM training iterations (Algorithm 3) |
| `SEQUENCE_LENGTH` | 125 | Input sequence length (Algorithm 4) |