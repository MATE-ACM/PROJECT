# MATE Experts (Open-Source Subset)

This repository contains a standalone subset of our multimodal emotion recognition expert framework. It covers expert definitions, dataset configuration, feature-based training, calibration, and post-hoc analysis for several benchmarks.

At this stage, the repository focuses on the expert side of the pipeline:

- dataset configuration via YAML
- expert configuration via YAML
- expert registry / model construction
- feature-based dataloading
- training, calibration, and analysis scripts

The main MATE framework will be released and refined incrementally in future updates.

---

## Included datasets

This repository currently contains three dataset-specific code snapshots:

- `iemocap/`
- `cremad/` (CREMA-D)
- `ravdess/`

Each directory is self-contained and follows the same high-level workflow.

### Notes

- **IEMOCAP-4 and IEMOCAP-6 use the same codebase.** In practice, you only need to switch the label setting from 4-way to 6-way.
- Due to historical reasons, some earlier RAVDESS experiments inherited naming conventions from older CREMA-D configurations. The open-source version has been cleaned up, and future releases will continue to unify naming under the main MATE framework.

---

## Repository structure

A typical dataset directory looks like this:

```text
<dataset>/
├── configs/
│   ├── datasets/
│   │   └── <dataset>.yaml
│   └── experts/
│       └── *.yaml
├── data/
│   ├── extract_audio/
│   ├── extract_video/
│   └── extract_txt/          # only where applicable
├── scripts/
│   ├── train.py
│   ├── calibrate.py
│   └── analyze_expert.py
├── src/ser/
│   ├── data/
│   ├── experts/
│   ├── losses.py
│   ├── metrics.py
│   ├── calibration.py
│   └── train_utils.py
└── run_click.py
```

---

## Training pipeline overview

The training flow is configuration-driven.

### 1. Prepare manifests and extracted features

The framework expects a manifest file (JSONL) plus pre-extracted modality features.

Typical examples include:

- audio features from WavLM / Whisper
- video features from HSEmotion / OpenFace
- text features from RoBERTa-like encoders

Each sample in the manifest usually contains at least:

- `utt_id`
- label information
- split/group metadata needed by the protocol
- optional transcript / speaker / session information, depending on dataset

Feature files are then stored by utterance ID and loaded by the expert configuration.

### 2. Configure the dataset

Each dataset has a dataset YAML under `configs/datasets/`.

This file defines:

- dataset name
- manifest path
- label set
- split protocol
- some compatibility placeholders for audio/video settings

For IEMOCAP, the same code supports both 4-way and 6-way classification through the dataset YAML.

Example:

```yaml
label_scheme: 4way
```

or

```yaml
label_scheme: 6way
```

The training script reads this switch and automatically updates the effective manifest path, labels, and number of classes.

### 3. Configure the expert

Each expert is defined by one YAML under `configs/experts/`.

An expert YAML usually specifies:

- `exp_name`
- `expert.type`
- feature root(s)
- input dimensions
- temporal limits / padding rules
- model hyperparameters
- optimizer and scheduler settings
- loss configuration

For example, a fusion expert may define:

- `audio_feat_root`
- `video_feat_root`
- `audio_input_dim`
- `video_input_dim`
- `num_classes`

### 4. Register the expert implementation

Model implementations are registered in `src/ser/experts/` through the expert registry.

The high-level rule is:

1. implement the model module
2. register it with the expert registry
3. reference the registered type in the expert YAML

This is what makes the framework YAML-driven: the YAML decides which expert implementation is built at runtime.

### 5. Train / calibrate / analyze

The main orchestration entry is usually `run_click.py` inside each dataset directory.

It does the following:

1. load dataset YAML
2. load selected expert YAML(s)
3. run cross-validation training fold by fold
4. optionally calibrate logits
5. optionally generate analysis artifacts

Outputs are saved to a run directory such as:

```text
runs/<exp_name>/fold_<k>/
```

Typical outputs include:

- checkpoints
- split records
- validation logs
- saved logits / predictions
- final metrics
- calibration reports
- reliability plots
- analysis outputs

---

## Minimal usage

### IEMOCAP

```bash
cd iemocap
python run_click.py
```

If you want to train a single expert/fold directly:

```bash
python scripts/train.py \
  --config wavlm_txt_lmf.yaml \
  --dataset configs/datasets/iemocap.yaml \
  --label_scheme 4way \
  --fold 0
```

### CREMA-D

```bash
cd cremad
python run_click.py
```

### RAVDESS

```bash
cd ravdess
python run_click.py
```

---

## How to select experts

The simplest way is to edit the `expert_yamls` list in each dataset's `run_click.py`.

For example, enable one or more YAMLs in:

```python
CONFIG = {
    "expert_yamls": [
        "fusion_av_gated.yaml",
        "cremad_fusion_wavlm_hsemotion_xattn.yaml",
    ]
}
```

This is the easiest batch-training entry point when reproducing multiple experts.

---

## Dataset-specific notes

### IEMOCAP

- IEMOCAP-4 and IEMOCAP-6 share the same code.
- You only need to switch the effective label scheme.
- The manifest and label list are selected from the dataset YAML.

### RAVDESS

- Some historical experiment files used legacy CREMA-D-style naming.
- The open-source version keeps the runnable structure while progressively cleaning these historical conventions.

### Text features for IEMOCAP

The IEMOCAP text feature extraction script was built around a workflow that previously reused components from MERBench-related processing. In this repository we keep only the subset needed to reproduce the local training pipeline.

---

## Reproducibility notes

This repository is intended as a clean and reusable open-source subset of a larger internal framework.

Before large-scale reproduction, please make sure to:

1. prepare dataset manifests on your own machine
2. extract features into local feature directories
3. update dataset/expert YAMLs so they point to your local manifests and feature folders
4. verify label spaces and fold settings per dataset

Because feature extraction and local dataset organization can vary by environment, exact end-to-end reproduction may still require small path and preprocessing adjustments.

---

## Requirements

### Hardware & Environment
To ensure full reproducibility and avoid out-of-memory (OOM) errors during feature extraction or training, the following hardware and environment setup was used and is recommended:
- **OS:** Windows 11
- **CPU:** AMD Ryzen 5 5600 6-Core Processor @ 3.50 GHz
- **RAM:** 32 GB
- **GPU:** NVIDIA RTX 5060 Ti (16GB VRAM）
- **CUDA Toolkit:** 12.8
- **PyTorch:** 2.8.0

### Python Dependencies
A baseline `requirements.txt` is provided at the repository root.

It is designed as a practical starting point for running training, calibration, and most feature extraction utilities. If you need strict environment-level reproducibility, you can additionally export a fully pinned environment from your own machine after validation.



## Ongoing development

This repository is only one part of the full project.

We will continue to gradually develop and release the main components of the MATE framework in future updates, including further cleanup, unification of dataset conventions, and more complete training/inference support.



## Acknowledgements

Some components in this repository were adapted from or inspired by the public MERBench / MERTools codebase for multimodal emotion recognition. In particular, parts of the text-feature processing adn Vedio-feature oricessing and some utility workflows were refactored to fit the configuration and training pipeline used in this repository.

We thank the authors of MERBench / MERTools for making their code publicly available.