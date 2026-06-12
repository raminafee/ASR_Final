# Automatic Mispronunciation Detection & Speech Intelligibility Assessment in Arabic-Speaking Children

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Wav2Vec2--Large--XLSR--53--Arabic-yellow)](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic)

An advanced clinical-linguistic AI framework designed for the automated, objective screening of **Speech Sound Disorders (SSDs)** and speech intelligibility in Arabic-speaking pediatric cohorts. This repository hosts a dual-paradigm computational ecosystem evaluating **End-to-End Deep Fine-Tuning** against a **Hybrid Handcrafted Bio-Acoustic Machine Learning Pipeline**.

---

## 📋 Project Overview

Traditional diagnostic assessment of SSDs in children relies on manual, subjective clinical transcriptions, which are highly vulnerable to intra-examiner variability and cause diagnostic bottlenecks in resource-constrained medical or educational environments. 

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
  * **Deletion (D):** Complete omission of necessary articulatory phonemes (e.g., omitting the rolling liquid $/r/$ in "ورق" $ightarrow$ "واق").
  * **Substitution (S):** Replacing standard Arabic phonemes due to motoric/cognitive constraints (e.g., swapping the uvular stop $/q/$ for the velar stop $/k/$ in "صديق" $ightarrow$ "صديك").

### 3. Speaker-Independent Partitioning
To guarantee generalization and prevent the models from memorizing speaker-specific fundamental frequencies ($F_0$) or domestic background acoustics, a strict speaker-independent boundary was enforced:
* **Training Set:** 1,375 utterances mapped to 11 isolated speakers (IDs: 2, 3, 6, 7, 8, 9, 10, 11, 13, 14, 15).
* **Validation Set:** 247 utterances mapped to Speaker ID: 5 and cohorts for hyperparameter tuning.
* **Test Set:** 378 utterances mapped to completely distinct, blind speakers reserved exclusively for final evaluation reporting.

---

## 🏗️ Core Methodological Frameworks

### Framework 1: The Hybrid Bio-Acoustic Pipeline
* **Deep Feature Extraction:** Extracts the activation vectors from the final transformer context block of a pre-trained, frozen `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` network, yielding $\mathbf{X}_{	ext{deep}} \in \mathbb{R}^{1024}$.
* **Handcrafted Micro-Acoustic Biomarkers:** Computes cycle-to-cycle physical glottal variations via sub-harmonic auto-correlation tracking:
  * **Local Jitter (%):** Measures short-term frequency perturbations of $F_0$ to capture vocal fold vibration instability or pediatric dysphonia.
  * **Absolute Shimmer (dB):** Quantifies short-term cycle-to-cycle peak amplitude variability, correlating with structural glottal friction and hoarseness.
  * **Harmonic-to-Noise Ratio (HNR):** Measures the balance between periodic acoustic energy and turbulent noise caused by incomplete vocal tract closures. A collapsing HNR explicitly flags severe phonetic substitutions or unvoiced distortions.
* **Two-Stage Shallow Classification:** Combines the features ($\mathbf{X}_{	ext{deep}} + \mathbf{X}_{	ext{bio}}$). An XGBoost algorithm evaluates the Split Information Gain to discard redundant dimensions, passing the **Top 100 Golden Features** to an optimized **Support Vector Machine (SVM)** utilizing a Non-linear Radial Basis Function (RBF) Kernel.

### Framework 2: The Fine-Tuned End-to-End Pipeline
* **Architecture Mutation:** Strips away the Connectionist Temporal Classification (CTC) layer. Injects a custom **Sequence Classification Head** (Dropout layer with $p=0.1$ feeding into a Dense Linear Layer mapping to `num_labels=2`).
* **Optimization:** Propagates gradients backwards through all encoder and self-attention blocks using Cross-Entropy Loss minimizations. This forces internal attention heads to structurally re-align latent weight configurations specifically for pediatric formant characteristics.
* **Character Error Rate (CER) Automation:** Deploys a parallel Wav2Vec2 CTC transcription model to compute normalized CER. This replaces standard Word Error Rate (WER) because children alter individual phonemes within isolated words (e.g., "صديق" $ightarrow$ "صديك" results in an explicit CER of 25.0%).

---

## 🔧 Preprocessing Pipeline

1. **Audio Conditioning:**
   * **Resampling:** All waveforms are forced to mono-channel 16 kHz to match native feature encoder fields.
   * **Amplitude Normalization:** Scales waveforms between $[-1.0, 1.0]$ to cancel out differences in microphone gains and recording distances.
   * **Voice Activity Detection (VAD):** Adaptive thresholding strips leading/trailing silence, isolating the active articulatory window.
2. **Linguistic Normalization:**
   * **Removal of Diacritics:** Systematically purges short vowels, vowel marks, and nunation (Fathah, Dammah, Kasrah, Sukun, Tanween) using regular expressions to prevent artificial string distance penalties.
   * **Orthographic Normalization:** Standardizes structural variants of Arabic characters (collapsing Alif `أ, إ, آ` into bare `ا`, Alif Maqsura `ى` into `ي`, and Ta Marbuta `ة` to `ه`).
   * **Purging:** Cleanses all non-Arabic text tokens, punctuation, and extraneous spaces.

---

## 📈 Experimental Results & Performance Matrix

The frameworks were thoroughly evaluated across the entire 2,000-sample matrix, producing distinct, highly stable computational behavior.

### 1. Fine-Tuning Loss Convergence
The empirical training log records indicate smooth, optimal convergence across 3 Epochs without overfitting:
* **Epoch 1:** Training Loss = 2.212494 | Validation Loss = 0.651872
* **Epoch 2:** Training Loss = 1.571686 | Validation Loss = 0.537741
* **Epoch 3:** Training Loss = 1.475045 | Validation Loss = 0.474773

### 2. Comprehensive Performance Metrics Breakdown

| Proposed System Paradigm | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fine-Tuned Wav2Vec2** | **86.75%** | **90.44%** | 73.88% | **95.00%** | **81.32%** | **0.9423** |
| **Advanced Filtered SVM** | 80.75% | 70.00% | **87.18%** | 76.63% | 0.7800 | N/A |
| **Hybrid XGBoost** | 79.75% | 77.00% | 69.00% | 86.88% | 0.7300 | N/A |

### 3. Mathematical Analysis of Confusion Matrices

$$\mathbf{M}_{	ext{XGBoost}} = egin{bmatrix} 212 & 32 \ 49 & 107 \end{bmatrix}$$
* *Interpretation:* High False Negatives (49 children with disorders missed) severely limit its individual clinical deployment, restricting its individual recall to 69.00%.

$$\mathbf{M}_{	ext{SVM}} = egin{bmatrix} 187 & 57 \ 20 & 136 \end{bmatrix}$$
* *Interpretation:* By explicitly manipulating the classification decision boundary from $	au = 0.50$ down to $	au_{	ext{SVM}} = 0.30$, the SVM became highly sensitive to pathological anomalies. It compressed missed cases from 49 to only 20, maximizing clinical safety with an outstanding **Recall of 87.18%**.

$$\mathbf{M}_{	ext{Fine-Tuned}} = egin{bmatrix} 1158 & 61 \ 204 & 577 \end{bmatrix}$$
* *Interpretation:* Out of 1,219 true normal instances, it correctly labels 1,158 cases, achieving an exceptional **Specificity of 95.00%**. It accurately pinpoints 577 speech disorder samples, yielding an unmatched **Precision of 90.44%** due to an exceptionally low False Positive rate (61 cases).

---

## 🔬 Applied Field Case Study: Live Out-of-Sample Test

To validate real-world out-of-sample generalization, 5 raw `.wav` recordings from an independent live pediatric screening folder (`Test_Child`) were passed through the production pipeline.

### Granular Patient Analytics Dashboard

| File Name | Target Word | Child Speech | Realized CER | Acoustic Bio-Markers (Jitter / Shimmer / HNR) | XGBoost Prediction | SVM Prediction | Fine-Tuned Wav2Vec2 | Final Diagnosis Status |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `حصان.wav` | حصان | حصان | 0.0% | 5.042% / 34.076% / 0.90 dB | Normal (14.1%) | Normal (10.3%) | Normal (11.8%) | ✅ **Normal Speech** |
| `مطار.wav` | مطار | مطاعين | 75.0% | 8.495% / 11.107% / -4.50 dB | Disorder (98.5%) | Disorder (91.4%) | Disorder (80.4%) | ⚠️ **Disorder Detected** |
| `صديق.wav` | صديق | صديك | 25.0% | 1.546% / 26.458% / 0.57 dB | Disorder (96.5%) | Disorder (91.4%) | Disorder (74.3%) | ⚠️ **Disorder Detected** |
| `ورق.wav` | ورق | واق | 33.3% | 1.500% / 29.804% / -1.53 dB | Disorder (94.4%) | Disorder (91.4%) | Disorder (66.1%) | ⚠️ **Disorder Detected** |
| `معلم.wav` | معلم | مو علم | 100.0% | 1.107% / 11.997% / 0.28 dB | Disorder (95.2%) | Disorder (91.4%) | Disorder (78.9%) | ⚠️ **Disorder Detected** |

### Interpretation of Performance Phenotypes
* **`حصان.wav` (Normal Baseline):** Perfect convergence across all frameworks. The Fine-Tuned Wav2Vec2 outputs a minimal disorder probability of 11.8%, aligning tightly with a CER of 0.0%.
* **`مطار.wav` (Severe Substitution & Insertion):** The child heavily distorts the word into "مطاعين". The catastrophic loss of signal periodic structure is captured by a highly negative HNR of -4.50 dB and an elevated Jitter of 8.495%. All models flag the anomaly with high confidence.
* **`صديق.wav` (Linguistic Substitution Phenotype):** The child swaps the standard uvular stop for a velar stop ("صديك"), a classic pediatric SSD pattern resulting in a 25.0% CER. The Fine-Tuned Wav2Vec2 easily flags this pathological pattern at 74.3% confidence.
* **`ورق.wav` (Phoneme Deletion Phenotype):** The child completely omits the liquid liquid consonant $/r/$, producing "واق". The missing phoneme drops the acoustic HNR to a negative -1.53 dB. The Fine-Tuned Wav2Vec2 successfully registers the deletion at 66.1% confidence.

---

## 🏛️ Comprehensive Engineering Justifications

1. **The Handcrafted vs. End-to-End Representation Justification:**
   When a pre-trained Wav2Vec2 network remains frozen (Framework 1 baseline), its attention weights remain optimized for standard adult language data, causing standalone deep feature accuracy to hover around random guessing levels (48%–52%). However, full end-to-end fine-tuning updates the neural transformer network's parameters. This allows the system to construct a specialized internal latent representation optimized for the high pitches and formant characteristics of children's voices, capturing Arabic acoustic transitions without human feature engineering.
2. **The Clinical Sensitivity Trade-Off Justification:**
   In medical screening, missing an affected child (a False Negative) is significantly more critical than flagging a healthy child for secondary clinical review (a False Positive). By extracting explicit micro-acoustic metrics (Jitter, Shimmer, HNR), the hybrid framework gains direct access to physical vocal fold vibration data and aerodynamic turbulence. When paired with an optimized decision boundary ($	au_{	ext{SVM}} = 0.30$), the SVM acts as an excellent clinical screening engine, catching subtle anomalies that end-to-end models might smooth over in pursuit of higher accuracy.

---

## 🚀 Standalone Production Deployment Strategy

Ultimately, the **Fine-Tuned Wav2Vec2 model is selected for standalone production deployment**. 

### Why Bypassing the Hybrid Models is Crucial for Production:
* **Elimination of Structural Overhead:** Deploying the hybrid pipeline requires a complex, multi-stage online runtime architecture where handcrafted metrics (Jitter, Shimmer, HNR) must be continuously tracked, calculated via sub-harmonic alignment, and filtered using XGBoost on the fly. This creates substantial computational latency and increases software maintenance risks.
* **Lightweight and Scalable:** The Fine-Tuned Wav2Vec2 model acts as a unified, single-stage computational engine. It processes the raw digital audio signal directly, completely eliminating the need for runtime feature engineering or external alignment blocks.
* **Dynamic Sensitivity Tuning:** Any relative drop in the fine-tuned model's baseline recall can be completely mitigated in production by adjusting and optimizing its classification decision threshold ($	au$). This dynamic software modification elevates clinical screening sensitivity instantly without introducing any runtime pipeline bottlenecks or architectural complexity, delivering a maintenance-free, fully scalable diagnostic engine.

---

## 👥 Authors & Research Affiliation
* **Rami Nafee**
* **Ibrahim Abu Zaid**
* **Ahmed Alnahhal**

*PhD Candidates, Department of Computer Engineering, Islamic University - Gaza, Palestine*
