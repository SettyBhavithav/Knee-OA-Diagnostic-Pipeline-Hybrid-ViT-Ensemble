# Hybrid Deep Learning Ensemble with Ordinal-Aware Loss for Knee Osteoarthritis Severity Grading

[![IEEE Paper](https://img.shields.io/badge/Paper-IEEE%20Format-blue.svg)](manuscript/knee_osteoarthritis_paper.tex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

An end-to-end, clinically validated, leakage-free deep learning framework for automated Knee Osteoarthritis (KOA) severity grading from radiographic X-ray images. This repository integrates a **triple-backbone hybrid architecture (ConvNeXt-Base + Swin Transformer + DINO ViT)** with ordinal-aware optimization, GPU dual-pass test-time augmentation (TTA), and a comprehensive multi-branch explainable AI (XAI) suite.

---

## Key Performance Benchmarks (GPU Dual-Pass TTA, N=4,008)

Evaluated on an independent, combined evaluation cohort of **4,008 images** (N=9,786 total dataset across 4,130 unique patients) with **verified zero patient-level leakage**:

| Model Variant | Accuracy (%) | Precision (%) | Recall (%) | Macro F1 (%) | QWK | Macro AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ConvNeXt-Base (Local Texture) | 87.52 | 88.07 | 90.27 | 89.05 | 0.800 | 0.966 |
| Swin Transformer (Global Context) | 84.26 | 83.00 | 87.28 | 84.35 | 0.749 | 0.948 |
| DINO ViT (Structural Priors) | 90.22 | 92.08 | 92.98 | 92.52 | 0.842 | 0.974 |
| **Hybrid Fusion Model** | **90.92** | **92.39** | **93.17** | **92.70** | **0.853** | **0.980** |
| **Soft-Voting Ensemble** | **90.54** | **92.02** | **92.80** | **92.28** | **0.847** | **0.980** |

### Clinical Safeguards & Key Achievements
- **100.00% Severe Class Sensitivity:** All 122 Severe test cases (N=4,008) correctly identified with **zero extreme grade reversals** (0 Severe misclassified as Healthy).
- **Quadratic Weighted Kappa (QWK):** **0.853**, indicating **almost perfect** ordinal agreement according to the Landis-Koch scale.
- **Verified Leakage-Free Splitting:** Strict patient-level partition with zero subject overlap across splits.

---

## Architecture Overview

`
                          [ Input Radiograph (224x224) ]
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  [ ConvNeXt-Base ]             [ Swin Transformer ]         [ DINO ViT-Base ]
 (Hierarchical Texture)        (Shifted-Window Context)    (Self-Supervised Structural)
    1024-dim Features            1024-dim Features            768-dim Features
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        ▼
                         [ Late Feature Concatenation ]
                              (2816-dimensional)
                                        ▼
                  [ Multi-Layer Head: FC(1024) -> FC(512) ]
                                        ▼
                     [ Softmax Output: 3 Severity Tiers ]
                    (Healthy: 0-1, Moderate: 2-3, Severe: 4)
`

---

## Repository Directory Structure

`
Knee-OA-Diagnostic-Pipeline-Hybrid-ViT-Ensemble/
├── manuscript/
│   ├── knee_osteoarthritis_paper.tex   # IEEEtran LaTeX source code (55 inline references)
│   ├── fig2.jpeg                       # Fig 2: Confusion Matrices
│   ├── fig3.jpeg                       # Fig 3: ROC Curves (Dual-Pass TTA)
│   ├── fig4.jpeg                       # Fig 4: Multi-Method Explainability Visualizations
│   └── fig5.jpeg                       # Fig 5: LIME Superpixel Attributions
│
├── notebooks/
│   └── knee_osteoarthritis_xai_and_evaluation.ipynb  # Primary GPU pipeline & XAI notebook
│
├── src/                                # Source Code & Execution Scripts
│   ├── knee_osteoarthritis_xai_and_evaluation.py    # Main evaluation script
│   ├── verify_datasplit_leakage_free.py              # Patient-level leakage audit tool
│   ├── run_statistical_tests.py                      # McNemar, Bootstrap CI, DeLong tests
│   ├── apply_tta_to_koa.py                           # TTA execution script
│   └── cache_gpu_preds.py                            # GPU probability caching utility
│
├── figures/                            # High-Resolution Publication Figures
│   ├── fig2.jpeg
│   ├── fig3.jpeg
│   ├── fig4.jpeg
│   └── fig5.jpeg
│
├── reports/
│   └── q1_journal_comprehensive_report.md            # Comprehensive audit & validation report
│
├── dataset/
│   └── README.md                                     # Dataset acquisition & split instructions
│
├── weights/
│   └── README.md                                     # Checkpoint storage instructions
│
├── .gitignore
└── README.md
`

---

## Manuscript Compilation

To compile the paper into PDF format:
`ash
cd manuscript/
pdflatex knee_osteoarthritis_paper.tex
pdflatex knee_osteoarthritis_paper.tex
`
*(All 55 references are embedded inline inside knee_osteoarthritis_paper.tex, so running BibTeX is not required.)*
