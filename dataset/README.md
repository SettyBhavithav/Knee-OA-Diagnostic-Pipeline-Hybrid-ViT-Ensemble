# Dataset Directory Structure

Dataset images are organized into patient-split partitions:

- `train/`: Training images (2,889 unique patients)
- `val/`: Validation images (413 unique patients)
- `test/`: Independent test images (828 unique patients)
- `auto_test/`: External test images (1,526 unique patients)

> **Note**: Raw dataset images are ignored by `.gitignore` due to storage limits. Ensure images follow `[PatientID][L/R].png` filename format for automatic zero-leakage parsing.
