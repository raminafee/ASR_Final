# Automatic Mispronunciation Detection in Arabic-Speaking Children

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Wav2Vec2--XLSR--Arabic-yellow)](https://huggingface.co/elgeish/wav2vec2-large-xlsr-53-arabic)

A hybrid diagnostic framework for automatic mispronunciation detection in Arabic-speaking children using **Wav2Vec2-XLSR-53-Arabic** deep phonetic embeddings and classical **acoustic-prosodic features**.

---

## 📋 Overview

This project presents a screening tool for detecting speech sound disorders (SSDs) in Arabic-speaking children. The system combines:

- **Deep Phonetic Embeddings** from `wav2vec2-large-xlsr-53-arabic` with cosine similarity via Dynamic Time Warping (DTW)
- **Acoustic-Prosodic Features** (Fundamental Frequency, Intensity, Normalized Duration)
- **Random Forest Classifier** for diagnostic decision-making

The system evaluates 5 Arabic words targeting diverse **Makharij** (articulatory zones) and applies a **40% aggregate error rate threshold** for clinical referral.

---

## 🗂️ Dataset: ASMDD

The **Arabic Speech Mispronunciation Detection Dataset (ASMDD)** is a purpose-built clinical-linguistic corpus.

| Property | Specification |
|---|---|
| Total audio samples | 35 |
| Number of speakers | 7 |
| Words per speaker | 5 (multi-Makharij coverage) |
| Audio format | .wav, uncompressed, monaural, 16-bit PCM |
| Sampling rate | 16,000 Hz |
| Label classes | `S` (Correct Pronunciation), `_N` (Mispronunciation) |
| Validation strategy | Leave-one-speaker-out (7 folds) |

---

## 🏗️ System Architecture


---

## 🔧 Technical Stack

| Phase | Tool / Technology | Purpose |
|---|---|---|
| Model Loading | Hugging Face Transformers | Load `wav2vec2-large-xlsr-53-arabic` |
| Acoustic Analysis | Parselmouth (Praat) | Extract Pitch (F0) and Intensity |
| Phonetic Comparison | Cosine Similarity + DTW | Measure articulation accuracy |
| Classification | Scikit-Learn (Random Forest) | Diagnostic decision |
| Evaluation | Mean Error Rate (threshold: 40%) | Referral recommendation |

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager


## 👥 Authors
- Rami Nafee
- Ibrahim Abu Zaid
- Ahmed Alnahhal

PhD Students
 Department of Computer Engineering
 - Islamic University - Gaza, Palestine
