# Automatic Mispronunciation Detection & Speech Intelligibility Assessment in Arabic-Speaking Children
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Wav2Vec2--XLSR--Arabic-yellow)](https://huggingface.co/elgeish/wav2vec2-large-xlsr-53-arabic)

## 📋 Project Overview

Traditional diagnostic assessment of Speech Sound Disorders (SSDs) in children relies on manual, subjective clinical transcriptions, which are highly vulnerable to intra-examiner variability and cause diagnostic bottlenecks in resource-constrained medical or educational environments. 

This project delivers a high-throughput, objective diagnostic metric by evaluating and deploying two distinct paradigms:
1. **Framework 1 (Hybrid ML):** A pipeline utilizing frozen Wav2Vec2 layers for deep acoustic embeddings ($\mathbf{X}_{	ext{deep}} \in \mathbb{R}^{1024}$), combined with handcrafted physical micro-acoustic biomarkers (Local Jitter, Absolute Shimmer, HNR). Features are filtered via an XGBoost Split Information Gain ranker (Top 100 Golden Features) and classified through an RBF-Kernel SVM optimized for high clinical sensitivity ($	au_{	ext{SVM}} = 0.30$).
2. **Framework 2 (End-to-End Deep Fine-Tuning):** A fully mutated `Wav2Vec2-Large-XLSR-53-Arabic` architecture, stripped of its native CTC layer and injected with a specialized Sequence Classification Head. The entire network is fully backpropagated using Cross-Entropy Loss to dynamically self-calibrate to pediatric pitch ranges and distinctive Arabic phone boundaries.

---

## 🗂️ Dataset Architecture: Smart Arabic Speech Therapy Dataset

The frameworks are trained, validated, and blind-tested on the **Smart Arabic Speech Therapy Dataset**, a large-scale, high-fidelity clinical speech corpus.

### 1. Quantitative and Linguistic Distribution
* **Total Audio Recordings:** 2,000 distinct high-fidelity pediatric speech samples.
* **Word-Level Utterances:** 1,500 recordings targeting single isolated target words to pinpoint explicit phoneme articulations across diverse **Makharij** (articulatory zones).
* **Sentence-Level Utterances:** 500 continuous speech recordings to evaluate contextual fluency, co-articulation, and prosodic decay.
* **Speakers:** 16 distinct Arabic-speaking children.
* **Audio Format:** 16,000 Hz ($16	ext{ kHz}$) sampling rate, mono-channel, uncompressed PCM `.wav` format to perfectly match native transformer structural constraints.

### 2. Class Balance & Pathological Annotations
The corpus exhibits a highly realistic, clinically balanced 60/40 label split:
* **Normal Class (Symptom-Free):** 1,219 recordings (60.95%) providing the physiological baseline.
* **Disordered Class (Pathological):** 781 recordings (39.05%) explicitly annotated by Speech-Language Pathologists (SLPs) at the phoneme level for three core structural linguistic errors:
  * **Insertion (I):** Intruding non-existent phonemes into the target word matrix.
  * **Deletion (D):** Complete omission of necessary articulatory phonemes (e.g., omitting the rolling liquid $/r/$ in "ورق" $
ightarrow$ "واق").
  * **Substitution (S):** Replacing standard Arabic phonemes due to motoric/cognitive constraints (e.g., swapping the uvular stop $/q/$ for the velar stop $/k/$ in "صديق" $
ightarrow$ "صديك").

### 3. Speaker-Independent Partitioning
To guarantee generalization and prevent the models from memorizing speaker-specific fundamental frequencies ($F_0$) or domestic background acoustics, a strict speaker-independent boundary was enforced:
* **Training Set:** 1,375 utterances mapped to 11 isolated speakers (IDs: 2, 3, 6, 7, 8, 9, 10, 11, 13, 14, 15).
* **Validation Set:** 247 utterances mapped to Speaker ID: 5 and cohorts for hyperparameter tuning.
* **Test Set:** 378 utterances mapped to completely distinct, blind speakers reserved exclusively for final evaluation reporting.

---

## 👥 Authors & Research Affiliation
* **Rami Nafee**
* **Ibrahim Abu Zaid**
* **Ahmed Alnahhal**

*PhD Candidates, Department of Computer Engineering, Islamic University - Gaza, Palestine*
