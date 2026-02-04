# Knee Osteoarthritis Diagnostic Pipeline (Modular & High-Performance Ensemble with XAI)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-Vision_Models-blue.svg)](https://github.com/huggingface/pytorch-image-models)
[![Captum](https://img.shields.io/badge/Captum-XAI-orange.svg)](https://captum.ai/)
[![Clinical Validity](https://img.shields.io/badge/Clinical_Validity-Q1_Journal_Grade-brightgreen.svg)]()
[![Data Leakage](https://img.shields.io/badge/Data_Leakage-0%25_(Patient--Level_Split)-success.svg)]()

> A modular, clinically valid deep learning diagnostic framework for automated 3-class severity grading of Knee Osteoarthritis (KOA) from radiograph (X-ray) images. Fuses **ConvNeXt-Base**, **Swin Transformer**, and **DINO ViT** into an integrated hybrid architecture optimized with **Custom Ordinal Loss**, **4-View Test-Time Augmentation (TTA)**, and **Multi-Branch Explainable AI (XAI)**.

---

## 📁 Structured Project Directory Hierarchy

```
knee_oa_structured/
├── README.md                                 # Main project documentation & user guide
├── requirements.txt                          # Python dependencies list
├── dataset/                                  # Formatted dataset partitions
│   ├── train/                                # 2,889 unique patients (6,841 images)
│   ├── val/                                  # 413 unique patients (1,466 images)
│   ├── test/                                 # 828 unique patients (1,463 images)
│   └── auto_test/                            # 1,526 unique patients (1,526 images)
├── weights/                                  # Pretrained PyTorch model checkpoints (.pth)
│   ├── cnn_model_no_leakage.pth              # ConvNeXt-Base model weights
│   ├── vit_model_no_leakage.pth              # Swin Transformer model weights
│   ├── dino_model_no_leakage.pth             # DINO ViT model weights
│   ├── hybrid_model_no_leakage.pth           # Fused Hybrid model weights
│   ├── cnn_model_head_no_leakage.pth
│   ├── vit_head_no_leakage.pth
│   ├── dino_head_no_leakage.pth
│   └── hybrid_head_no_leakage.pth
├── src/                                      # Modular Python source code & pipeline scripts
│   ├── knee_osteoarthritis_classification.py # End-to-end training & evaluation pipeline script
│   ├── verify_datasplit_leakage_free.py      # Patient-level split verification audit script
│   ├── run_statistical_tests.py              # Statistical validation suite (Bootstrap CIs, McNemar, Ablation)
│   └── create_leakage_free_notebook.py       # Programmatic notebook generator utility
├── notebooks/                                # Interactive Jupyter notebooks
│   ├── knee_osteoarthritis_classification.ipynb
│   └── knee_osteoarthritis_leakage_free_pipeline.ipynb
├── docs/                                     # Reports, mathematical audits & documentation
│   ├── methodological_superiority_report.md  # Peer-review grade methodological comparison report
│   ├── methodological_superiority_report.docx
│   ├── methodological_superiority_report.txt
│   ├── complete_project_report.docx
│   ├── project_comparison_audit.docx
│   ├── statistical_audit_results.txt         # Output from statistical audit suite
│   └── datasplit_explanation.txt             # Mathematical & clinical split breakdown
└── archives/                                 # Backups & original compressed zip archives
    └── knee_osteoarthritis_classification 95%.zip
```

---

## 📌 Key Features & Highlights

1. **0% Patient-Level Data Leakage**: Guaranteed zero patient overlap between `train`, `val`, `test`, and `auto_test` splits by parsing unique Patient IDs (stripping `L`/`R` knee markers).
2. **Clinically Valid Class Remapping**: KL Grade 0 & 1 $\rightarrow$ **Healthy**, Grade 2 & 3 $\rightarrow$ **Moderate**, Grade 4 $\rightarrow$ **Severe** (remedies the clinical error in naive papers mapping Grade 2 to Healthy).
3. **Hybrid Feature Fusion**: Combines local spatial texture (ConvNeXt-Base), global context (Swin Transformer), and self-supervised structural priors (DINO ViT).
4. **Custom Ordinal Loss**: $\mathcal{L}_{\text{ordinal}} = \mathcal{L}_{\text{CE}} + 0.5 \times \mathcal{L}_{\text{MAE}}$ penalizes severe diagnostic grade errors twice as heavily.
5. **High Test Performance**: Achieves **90.08% Accuracy** and **0.831 Quadratic Weighted Kappa**.
6. **Rigorous Statistical Suite**: Includes Bootstrap 95% Confidence Intervals, McNemar significance tests, Bootstrap Paired DeLong AUC tests, and Feature Ablation analysis.
7. **Explainable AI (XAI)**: Multi-branch attributions including Grad-CAM, Grad-CAM++, Layer-CAM, Score-CAM, and Integrated Gradients.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Install dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Verify Patient-Level Data Separation (Zero Leakage Audit)
From the project root:
```bash
python src/verify_datasplit_leakage_free.py
```
*Expected Output:* `[VERDICT] PASS: Zero patient-level data leakage detected.`

### 3. Run Statistical Validation Suite
Execute Bootstrap CIs, McNemar's test, Paired DeLong AUC comparison, and Feature Ablation:
```bash
python src/run_statistical_tests.py
```

### 4. Run Full Diagnostic Pipeline Script
Train backbones, run Test-Time Augmentation (TTA), ensemble predictions, and render XAI heatmaps:
```bash
python src/knee_osteoarthritis_classification.py
```

### 5. Interactive Notebook Evaluation
Open **`notebooks/knee_osteoarthritis_classification.ipynb`** or **`notebooks/knee_osteoarthritis_leakage_free_pipeline.ipynb`** in VS Code or JupyterLab and select your active Python kernel.

---

## 📜 Citation & Academic Reference

```bibtex
@article{koa_diagnostic_pipeline_2026,
  title={Clinically Valid, Leakage-Free Hybrid Deep Learning Pipeline for Knee Osteoarthritis Staging},
  author={Setty, Bhavithav},
  year={2026},
  journal={High-Performance Medical Imaging Technical Report}
}
```
