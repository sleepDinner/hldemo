# AGENTS.md

This file defines the long-term development rules for this repository. All future coding, refactoring, experiments, and documentation work should follow these rules unless the user explicitly overrides them.

## 1. Project Background

This project is a PyTorch project for image local manipulation detection / image manipulation localization.

The core task is:

- Input: one image.
- Output: a pixel-level tampering probability map and a binary tampering mask.

The project should be developed as a complete research codebase suitable for local development, server training, reproducible experiments, ablation studies, robustness evaluation, and paper-oriented result analysis.

## 2. Engineering Requirements

- Use PyTorch as the deep learning framework.
- Keep the code modular and maintainable.
- Put all paths, hyperparameters, model options, data settings, training settings, and experiment settings in YAML configuration files.
- Do not hardcode absolute paths in Python code, shell scripts, or config templates.
- Separate training, validation, testing, and inference logic.
- Support checkpoint resume for interrupted training.
- Save both the best model and the last model.
- Save training logs, metric curves, and predicted mask visualization images.
- Fix all random seeds used by Python, NumPy, PyTorch, CUDA, dataloaders, and augmentation logic whenever applicable.
- After each new feature, check imports, basic run commands, and file paths.

## 3. Dataset Requirements

- Support paired image and mask loading.
- The dataset implementation must be compatible with common image manipulation localization datasets, including but not limited to:
  - CASIA
  - Columbia
  - COVERAGE
  - NIST16
  - IMD2020
- Dataset root paths, image directories, mask directories, split files, and preprocessing settings must be specified through YAML configuration files.
- Dataset code should avoid dataset-specific assumptions unless isolated in clearly named adapters or parsers.
- Image-mask alignment must be validated carefully, especially after resizing, cropping, padding, or augmentation.

## 4. Model Requirements

The model architecture should be suitable for image local manipulation detection and should include the following design directions:

- RGB main branch.
- Noise / residual / high-frequency auxiliary branch.
- Local CNN feature extraction for texture, boundary, and local artifact cues.
- Global Mamba or Transformer modeling for long-range dependency and semantic-context consistency.
- Multi-scale decoder for dense pixel-level localization.
- Boundary refinement module for sharper tampering masks.

Model code should be split into clear modules, such as backbones, residual or frequency extraction, fusion blocks, decoders, boundary heads, and final prediction heads.

## 5. Experiment Requirements

The project must support:

- Main experiments.
- Ablation experiments.
- Cross-dataset testing.
- Robustness testing.
- Visualization result saving.

Robustness testing should cover practical post-processing operations when possible, such as JPEG compression, resizing, blur, noise, and other common image transmission degradations.

## 6. Evaluation Metrics

The project must implement the following metrics:

- F1
- IoU
- AUC
- Precision
- Recall
- MCC
- FPR

Metric implementations must be reusable for validation, testing, cross-dataset evaluation, and robustness evaluation. Threshold-dependent and threshold-independent metrics should be clearly separated.

## 7. Prohibited Practices

- Do not fabricate experimental results.
- Do not write pseudo-code that cannot run.
- Do not delete existing files casually.
- Do not introduce excessive or unnecessary dependencies.
- Any new dependency must be justified with a clear reason.
- Do not generate an oversized codebase in one step. Prefer incremental, modular implementation.
- Do not mix unrelated refactors with the requested task.
- Do not silently change experiment protocols, metric definitions, or dataset splits.

## 8. Completion Standard

After completing any task, report:

- Which files were modified.
- How to run or verify the change.
- What output is expected.
- Whether the user needs to provide dataset paths or server information.

If a task cannot be fully completed, clearly state the blocker and the minimal information or resource needed to continue.
