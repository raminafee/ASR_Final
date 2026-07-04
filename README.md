# Automatic Mispronunciation Detection and Speech Intelligibility Assessment in Arabic-Speaking Children
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Wav2Vec2--XLSR--Arabic-yellow)](https://huggingface.co/elgeish/wav2vec2-large-xlsr-53-arabic)

## 📋 Project Overview

Traditional diagnostic assessment of Speech Sound Disorders (SSDs) in children relies on manual, subjective clinical transcriptions, which are highly vulnerable to intra-examiner variability and cause diagnostic bottlenecks in resource-constrained medical or educational environments. 

**Highlights:**

• A comprehensive comparison between end-to-end, hybrid, and conventional acoustic baseline models for Arabic child mispronunciation detection.

• A fine-tuned Wav2Vec2 model achieved the highest overall performance with 91.80% accuracy and an AUC of 0.9864.

• Hybrid models combining Wav2Vec2 embeddings with clinical acoustic features outperformed conventional handcrafted acoustic baselines.

• Conventional MFCC and eGeMAPS baselines showed substantially lower performance under the official speaker-independent evaluation protocol.

• Deep contextual speech representations demonstrated superior generalization to previously unseen child speakers

This project delivers a high-throughput, objective diagnostic metric by evaluating and deploying two distinct paradigms:
1. **Framework 1 (Hybrid ML):** A pipeline utilizing frozen Wav2Vec2 layers for deep acoustic embeddings (*X_deep* in *R^1024*), combined with handcrafted physical micro-acoustic biomarkers (Local Jitter, Absolute Shimmer, HNR). Features are filtered via an XGBoost Split Information Gain ranker (Top 100 Golden Features) and classified through an RBF-Kernel SVM optimized for high clinical sensitivity.
2. **Framework 2 (End-to-End Deep Fine-Tuning):** A fully mutated `Wav2Vec2-Large-XLSR-53-Arabic` architecture, stripped of its native CTC layer and injected with a specialized Sequence Classification Head. The entire network is fully backpropagated using Cross-Entropy Loss to dynamically self-calibrate to pediatric pitch ranges and distinctive Arabic phone boundaries.

---

## 🗂️ Dataset Architecture: Smart Arabic Speech Therapy Dataset
🔗 **Dataset Repository:** [Access the dataset here](https://www.kaggle.com/datasets/abdelmonemhatem/tts-mispronunciation-detection)

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

---
## 📝 Technical Notes & Usage Guide

## ⚙️ Framework 1:

* **Notebook Execution:** The first framework was successfully executed locally using the notebook file named **`SVM_XGBoost_improve.ipynb`**. The notebook is structured into distinct cells, with dedicated markdown cells included to clarify specific methodological points and observations throughout the workflow.
* **Audio Directory Structure:** The dataset's audio recordings are located within a core directory named `audio/` which was downloaded from dataset link. 
* **Cell 1 (Pipeline & Model Training):** The first cell automatically inspects the actual content of the `audio/` directory, indexes the audio samples, extracts the micro-acoustic features, completes the data preprocessing, trains both models, and outputs the final accuracy metrics for both the **SVM** and **XGBoost** frameworks.
* **Cell 2 (private check):** The second cell executes the evaluation and testing pipeline specifically on a dedicated audio directory belonging to a single child, named `Test_Child/`.
* **Cell 3 (Interactive Child Testing Interface):** The subsequent cell provides a functional interface designed to evaluate new or arbitrary speech samples from any child. To test new samples, the respective audio files must be placed inside the `Test_Child/` directory before executing the cell.

## ⚙️ Framework 2: End-to-End Fine-Tuning Execution Details

* **Execution Environment:** Due to the high computational requirements of deep transformer optimization, this experiment was executed on **Google Colab** leveraging dedicated hardware acceleration (**CUDA GPU**).
* **Notebook Structure:** The architecture is deployed via the notebook named **`fine_tuning.ipynb`**, which is logically partitioned into functional execution cells.
* **Cell 1 (Dataset Diagnostics & Distribution):** The initial cell inspects the entire clinical corpus of **2,000 audio files**, automatically stratifying the samples into distinct pathological categories (Disordered vs. Normal classes).

* **Training and Convergence Log:**
  The full end-to-end backpropagation trained stably for **3 Epochs** across 300 optimization steps over approximately **18 minutes and 26 seconds**, demonstrating smooth loss decay:

  | Epoch | Training Loss | Validation Loss |
  | :---: | :---: | :---: |
  | 1 | 2.212494 | 0.651872 |
  | 2 | 1.571686 | 0.537741 |
  | 3 | 1.475045 | 0.474773 |

  Upon successful convergence, the final model shards were verified, assembled, and permanently serialized to Google Drive at: `/content/drive/MyDrive/ASRFiles/my_finetuned_wav2vec2_model`.

* **Inference Pipeline & Out-of-Sample Screening:**
  A subsequent dedicated cell loads the production-ready fine-tuned model directly from the Google Drive directory. This cell is built as an out-of-sample diagnostic utility. It automatically screens any targeted directory containing speech waveforms for a specific patient—configured by default to scan `/content/drive/MyDrive/ASRFiles/Test_Child`—and extracts the finalized algorithmic diagnosis and intelligibility assessment metrics.

### 📁 Directory Structure & Storage (Google Drive / Local)
🔗 **Google Drive ASRFiles Folder:** [Access the folder here](https://drive.google.com/drive/folders/1YZ8Mg05bqGoSaTWV38D24kRD895KxCQL?usp=sharing)

* **ASRFiles/** (Root Directory)
    * 📁 **audio/** — Contains the baseline clinical corpus (2,000 `.wav` files).
    * 📁 **Test_Child/** — Dedicated directory for out-of-sample patient screening and evaluation files.
    * 📁 **my_finetuned_wav2vec2_model/** — Serialized model checkpoints and weights saved after fine-tuning.
      
*PhD Candidates, Department of Computer Engineering, Islamic University - Gaza, Palestine*

