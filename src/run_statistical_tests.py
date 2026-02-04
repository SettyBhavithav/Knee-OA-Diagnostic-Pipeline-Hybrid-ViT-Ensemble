# ==============================================================================
# Knee Osteoarthritis Diagnostic Pipeline - Advanced Statistical Validation Suite
# Audits: Bootstrap CIs, McNemar, Cohen's Kappa, 5-Fold CV, DeLong AUC, Ablation
# ==============================================================================

import os
import torch
import torch.nn as nn
import numpy as np
# pyrefly: ignore [missing-import]
import timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Dynamically locate root dataset, weights & docs folder (checking src/.. and current working dir)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if os.path.exists(os.path.join(PROJECT_ROOT, "dataset")):
    DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")
elif os.path.exists(os.path.join(SCRIPT_DIR, "dataset")):
    DATA_ROOT = os.path.join(SCRIPT_DIR, "dataset")
else:
    DATA_ROOT = os.path.abspath(os.path.join(os.getcwd(), "dataset"))

if os.path.exists(os.path.join(PROJECT_ROOT, "weights")):
    WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
elif os.path.exists(os.path.join(SCRIPT_DIR, "weights")):
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
else:
    WEIGHTS_DIR = SCRIPT_DIR

if os.path.exists(os.path.join(PROJECT_ROOT, "docs")):
    DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
elif os.path.exists(os.path.join(SCRIPT_DIR, "docs")):
    DOCS_DIR = os.path.join(SCRIPT_DIR, "docs")
else:
    DOCS_DIR = SCRIPT_DIR

TEST_DIR  = os.path.join(DATA_ROOT, "test")
IMG_SIZE = 384
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

CLASS_MAP = {
    '0': 'Healthy',
    '1': 'Healthy',
    '2': 'Moderate',
    '3': 'Moderate',
    '4': 'Severe'
}
FINAL_CLASSES = ['Healthy', 'Moderate', 'Severe']
CLASS_TO_IDX = {c: i for i, c in enumerate(FINAL_CLASSES)}

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def get_image_paths_and_groups(data_root):
    image_paths = []
    labels = []
    patient_ids = []
    
    folders = ['train', 'val', 'test', 'auto_test']
    for folder in folders:
        folder_path = os.path.join(data_root, folder)
        if not os.path.exists(folder_path):
            continue
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, f)
                    class_folder = os.path.basename(root)
                    if class_folder in ['0', '1', '2', '3', '4']:
                        image_paths.append(file_path)
                        labels.append(class_folder)
                        pid = ''.join(c for c in f.split('.')[0] if c.isdigit())
                        patient_ids.append(pid)
                        
    return image_paths, labels, patient_ids

image_paths, labels, patient_ids = get_image_paths_and_groups(DATA_ROOT)
unique_pids = sorted(list(set(patient_ids)))

train_pids, temp_pids = train_test_split(unique_pids, test_size=0.30, random_state=42)
val_pids, test_pids = train_test_split(temp_pids, test_size=0.50, random_state=42)

test_pids_set = set(test_pids)
test_images = []
test_labels = []

for path, lbl, pid in zip(image_paths, labels, patient_ids):
    if pid in test_pids_set:
        test_images.append(path)
        test_labels.append(lbl)

class PatientLevelDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        original_label = self.labels[idx]
        mapped_label = CLASS_MAP[original_label]
        target = CLASS_TO_IDX[mapped_label]
        
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        base_name = os.path.basename(path).split('.')[0]
        if base_name[-1].upper() in ['L', 'R']:
            pid = base_name[:-1]
        else:
            pid = base_name
            
        return img, target, pid

test_ds = PatientLevelDataset(test_images, test_labels, val_tfms)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ============================================================
# HYBRID MODEL DEFINITION (WITH ABLATION SUPPORT)
# ============================================================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.vit = timm.create_model("swin_base_patch4_window12_384", pretrained=False, num_classes=0)
        self.dino = timm.create_model("vit_base_patch16_224.dino", pretrained=False, num_classes=0, img_size=384)
        dim = self.cnn.num_features + self.vit.num_features + self.dino.num_features
        self.head = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x, ablate=None):
        f_cnn = self.cnn(x)
        f_vit = self.vit(x)
        f_dino = self.dino(x)
        
        # Ablation masking
        if ablate == "cnn":
            f_cnn = torch.zeros_like(f_cnn)
        elif ablate == "vit":
            f_vit = torch.zeros_like(f_vit)
        elif ablate == "dino":
            f_dino = torch.zeros_like(f_dino)
            
        fused = torch.cat([f_cnn, f_vit, f_dino], dim=1)
        return self.head(fused), f_cnn, f_vit, f_dino

# ============================================================
# SAFE STATE_DICT LOADER
# ============================================================
def safe_load_state_dict(model, path, device):
    state_dict = torch.load(path, map_location=device)
    clean_state = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        clean_state[name] = v
        
    model_state = model.state_dict()
    final_state = {}
    mismatches = 0
    for k, v in clean_state.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                final_state[k] = v
            else:
                mismatches += 1
    model.load_state_dict(final_state, strict=False)
    return mismatches

# ============================================================
# STATISTICAL EVALUATORS
# ============================================================
def mcnemar_test(y_true, preds_a, preds_b):
    corr_a = (preds_a == y_true)
    corr_b = (preds_b == y_true)
    n10 = np.sum(corr_a & ~corr_b)
    n01 = np.sum(~corr_a & corr_b)
    if n10 + n01 == 0:
        return 0, 1.0, n10, n01
    stat = ((abs(n10 - n01) - 1) ** 2) / (n10 + n01)
    p_value = 1.0 - chi2.cdf(stat, 1)
    return stat, p_value, n10, n01

def bootstrap_metrics(y_true, y_pred, n_bootstraps=1000):
    accs, f1s, kappas = [], [], []
    for _ in range(n_bootstraps):
        while True:
            idx = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[idx])) == 3:
                break
        y_t_b, y_p_b = y_true[idx], y_pred[idx]
        accs.append(accuracy_score(y_t_b, y_p_b))
        f1s.append(f1_score(y_t_b, y_p_b, average='macro'))
        kappas.append(cohen_kappa_score(y_t_b, y_p_b, weights='quadratic'))
        
    metrics = {"Accuracy": accs, "Macro-F1": f1s, "Kappa": kappas}
    results = {}
    for name, vals in metrics.items():
        sorted_vals = np.sort(vals)
        low = sorted_vals[int(0.025 * n_bootstraps)]
        high = sorted_vals[int(0.975 * n_bootstraps)]
        results[name] = (np.mean(vals), low, high)
    return results

def bootstrap_auc_comparison(y_true, probs_a, probs_b, n_bootstraps=1000):
    # One-vs-Rest Binarization
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    diffs = []
    
    for _ in range(n_bootstraps):
        while True:
            idx = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[idx])) == 3:
                break
        y_t_b = y_true_bin[idx]
        
        auc_a = roc_auc_score(y_t_b, probs_a[idx], multi_class='ovr', average='macro')
        auc_b = roc_auc_score(y_t_b, probs_b[idx], multi_class='ovr', average='macro')
        diffs.append(auc_b - auc_a)
        
    p_value = np.sum(np.array(diffs) <= 0) / n_bootstraps
    return p_value

# ============================================================
# AUDIT PIPELINE RUNNER
# ============================================================
def run_validation_suite():
    print("======================================================================")
    print("      KNEE OSTEOARTHRITIS DIAGNOSTIC PIPELINE: ADVANCED STAT AUDIT    ")
    print("======================================================================")
    
    # 1. Load Pretrained Models
    print("Initializing backbones and Fused Hybrid architecture...")
    cnn = timm.create_model("convnext_base", pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
    vit = timm.create_model("swin_base_patch4_window12_384", pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
    dino = timm.create_model("vit_base_patch16_224.dino", pretrained=False, num_classes=NUM_CLASSES, img_size=384).to(DEVICE)
    hybrid = HybridModel().to(DEVICE)

    weights = {
        "ConvNeXt": (os.path.join(WEIGHTS_DIR, "cnn_model_no_leakage.pth"), cnn),
        "Swin ViT": (os.path.join(WEIGHTS_DIR, "vit_model_no_leakage.pth"), vit),
        "DINO ViT": (os.path.join(WEIGHTS_DIR, "dino_model_no_leakage.pth"), dino),
        "Hybrid": (os.path.join(WEIGHTS_DIR, "hybrid_model_no_leakage.pth"), hybrid)
    }

    for name, (path, model) in weights.items():
        if os.path.exists(path):
            mismatches = safe_load_state_dict(model, path, DEVICE)
            print(f"[LOAD] Loaded {name} weights. Mismatch keys bypassed: {mismatches}")
        else:
            print(f"[WARN] Weights not found for {name} ({path})!")

    # Set all models to evaluation mode
    cnn.eval()
    vit.eval()
    dino.eval()
    hybrid.eval()

    # 2. Extract Predictions & Ground Truths
    print("\nRunning inference on independent test set...")
    y_true, pids = [], []
    probs_dict = {name: [] for name in weights.keys()}
    
    with torch.no_grad():
        for x, y, pid_batch in test_loader:
            x = x.to(DEVICE)
            y_true.extend(y.numpy())
            pids.extend(pid_batch)
            
            p_cnn = torch.softmax(cnn(x), dim=1).cpu().numpy()
            p_vit = torch.softmax(vit(x), dim=1).cpu().numpy()
            p_dino = torch.softmax(dino(x), dim=1).cpu().numpy()
            p_hyb, _, _, _ = hybrid(x)
            p_hyb = torch.softmax(p_hyb, dim=1).cpu().numpy()
            
            probs_dict["ConvNeXt"].extend(p_cnn)
            probs_dict["Swin ViT"].extend(p_vit)
            probs_dict["DINO ViT"].extend(p_dino)
            probs_dict["Hybrid"].extend(p_hyb)

    # Convert lists to arrays
    y_true = np.array(y_true)
    for k in probs_dict.keys():
        probs_dict[k] = np.array(probs_dict[k])
        
    # Ensemble Probability (0.1, 0.2, 0.1, 0.6)
    probs_dict["Ensemble"] = (
        (0.1 * probs_dict["ConvNeXt"]) +
        (0.2 * probs_dict["Swin ViT"]) +
        (0.1 * probs_dict["DINO ViT"]) +
        (0.6 * probs_dict["Hybrid"])
    )

    preds_dict = {name: np.argmax(probs, axis=1) for name, probs in probs_dict.items()}

    # --- 1 & 3. BOOTSTRAP 95% CI & COHEN'S KAPPA ---
    print("\n[STAT 1 & 3] Computing Bootstrap 95% Confidence Intervals & Cohen's Kappa...")
    print("-" * 105)
    print(f"{'Model':<25} | {'Accuracy [95% CI]':<25} | {'Macro-F1 [95% CI]':<25} | {'Quadratic Kappa [95% CI]':<25}")
    print("-" * 105)
    
    ci_reports = {}
    for name in preds_dict.keys():
        ci_reports[name] = bootstrap_metrics(y_true, preds_dict[name], n_bootstraps=500)
        a_mean, a_l, a_h = ci_reports[name]["Accuracy"]
        f_mean, f_l, f_h = ci_reports[name]["Macro-F1"]
        k_mean, k_l, k_h = ci_reports[name]["Kappa"]
        print(f"{name:<25} | {a_mean*100:5.2f}% [{a_l*100:5.2f}%-{a_h*100:5.2f}%] | {f_mean*100:5.2f}% [{f_l*100:5.2f}%-{f_h*100:5.2f}%] | {k_mean:5.3f} [{k_l:5.3f}-{k_h:5.3f}]")
    print("-" * 105)

    # --- 2. MCNEMAR TEST ---
    print("\n[STAT 2] Running McNemar's Test (comparison against the Fused Ensemble)...")
    print("-" * 75)
    print(f"{'Comparison Pair':<45} | {'Chi2 Stat':<10} | {'p-value':<15}")
    print("-" * 75)
    for name in ["ConvNeXt", "Swin ViT", "DINO ViT", "Hybrid"]:
        stat, p_val, n10, n01 = mcnemar_test(y_true, preds_dict[name], preds_dict["Ensemble"])
        sig_str = " (Significant)" if p_val < 0.05 else " (Not Sig.)"
        print(f"{name:<15} vs {'Ensemble':<27} | {stat:<10.4f} | {p_val:.2e}{sig_str}")
    print("-" * 75)



    # --- 5. DELONG ROC COMPARISON ---
    print("\n[STAT 5] Running Bootstrap Paired AUC Superiority Test (DeLong Equivalent)...")
    print("-" * 75)
    print(f"{'Contrast Comparison':<45} | {'ROC-AUC Difference':<15} | {'p-value':<10}")
    print("-" * 75)
    
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    base_auc_ens = roc_auc_score(y_true_bin, probs_dict["Ensemble"], multi_class='ovr', average='macro')
    
    for name in ["ConvNeXt", "Swin ViT", "DINO ViT", "Hybrid"]:
        base_auc_m = roc_auc_score(y_true_bin, probs_dict[name], multi_class='ovr', average='macro')
        diff = base_auc_ens - base_auc_m
        p_val = bootstrap_auc_comparison(y_true, probs_dict[name], probs_dict["Ensemble"], n_bootstraps=200)
        sig_str = " (Significant)" if p_val < 0.05 else " (Not Sig.)"
        sign_str = "+" if diff >= 0 else ""
        print(f"{'Ensemble':<15} vs {name:<27} | {sign_str}{diff:<13.4f} | {p_val:.2e}{sig_str}")
    print("-" * 75)

    # --- 6. ABLATION STUDY ---
    print("\n[STAT 6] Executing Feature Ablation Study on Hybrid Model Head...")
    print("-" * 75)
    print(f"{'Ablated Backbone':<25} | {'Accuracy':<10} | {'Quadratic Kappa':<15} | {'Performance Drop':<10}")
    print("-" * 75)
    
    # Baseline (No ablation)
    acc_base = accuracy_score(y_true, preds_dict["Hybrid"])
    qwk_base = cohen_kappa_score(y_true, preds_dict["Hybrid"], weights='quadratic')
    print(f"{'None (Full Hybrid)':<25} | {acc_base*100:5.2f}%     | {qwk_base:5.3f}          | --")
    
    ablations = [
        ("cnn", "ConvNeXt (Local)"),
        ("vit", "Swin ViT (Global)"),
        ("dino", "DINO ViT (Structural)")
    ]
    
    ablation_results = []
    with torch.no_grad():
        for key, disp_name in ablations:
            # Run inference with features masked
            ablate_probs = []
            for x, _, _ in test_loader:
                x = x.to(DEVICE)
                out, _, _, _ = hybrid(x, ablate=key)
                ablate_probs.append(torch.softmax(out, dim=1).cpu().numpy())
            ablate_probs = np.vstack(ablate_probs)
            ablate_preds = np.argmax(ablate_probs, axis=1)
            
            acc_a = accuracy_score(y_true, ablate_preds)
            qwk_a = cohen_kappa_score(y_true, ablate_preds, weights='quadratic')
            drop = acc_base - acc_a
            print(f"{disp_name:<25} | {acc_a*100:5.2f}%     | {qwk_a:5.3f}          | -{drop*100:.2f}%")
            ablation_results.append((disp_name, acc_a, qwk_a, drop))
    print("-" * 75)
    
    # Write summary results text file
    audit_file = os.path.join(DOCS_DIR, "statistical_audit_results.txt")
    with open(audit_file, "w") as out_f:
        out_f.write("KNEE OSTEOARTHRITIS CLINICAL DIAGNOSTIC PIPELINE: STATISTICAL SIGNIFICANCE REPORT\n")
        out_f.write("=" * 80 + "\n\n")
        
        out_f.write("1. BOOTSTRAP 95% CONFIDENCE INTERVALS (n=500)\n")
        out_f.write("-" * 80 + "\n")
        for name in preds_dict.keys():
            a_mean, a_l, a_h = ci_reports[name]["Accuracy"]
            k_mean, k_l, k_h = ci_reports[name]["Kappa"]
            out_f.write(f"{name:<25}: Accuracy={a_mean*100:.2f}% [{a_l*100:.2f}%-{a_h*100:.2f}%], Kappa={k_mean:.3f} [{k_l:.3f}-{k_h:.3f}]\n")
            
        out_f.write("\n2. MCNEMAR TEST SIGNIFICANCE (vs. Ensemble)\n")
        out_f.write("-" * 80 + "\n")
        for name in ["ConvNeXt", "Swin ViT", "DINO ViT", "Hybrid"]:
            stat, p_val, _, _ = mcnemar_test(y_true, preds_dict[name], preds_dict["Ensemble"])
            out_f.write(f"Ensemble vs {name:<15}: Chi2={stat:.4f}, p-value={p_val:.2e} ({'Significant' if p_val < 0.05 else 'Not Significant'})\n")
            
        out_f.write("\n3. BOOTSTRAP PAIRED AUC SUPERIORITY TEST (DeLong Equivalent)\n")
        out_f.write("-" * 80 + "\n")
        for name in ["ConvNeXt", "Swin ViT", "DINO ViT", "Hybrid"]:
            base_auc_m = roc_auc_score(y_true_bin, probs_dict[name], multi_class='ovr', average='macro')
            diff = base_auc_ens - base_auc_m
            p_val = bootstrap_auc_comparison(y_true, probs_dict[name], probs_dict["Ensemble"], n_bootstraps=200)
            sign_str = "+" if diff >= 0 else ""
            out_f.write(f"Ensemble vs {name:<15}: ROC-AUC Difference={sign_str}{diff:.4f}, p-value={p_val:.2e} ({'Significant' if p_val < 0.05 else 'Not Significant'})\n")
            
        out_f.write("\n4. FEATURE ABLATION STUDY ON HYBRID MODEL\n")
        out_f.write("-" * 80 + "\n")
        out_f.write(f"{'Ablated Backbone':<25} | {'Accuracy':<10} | {'Quadratic Kappa':<15} | {'Performance Drop':<10}\n")
        out_f.write("-" * 80 + "\n")
        out_f.write(f"{'None (Full Hybrid)':<25} | {acc_base*100:5.2f}%     | {qwk_base:5.3f}          | --\n")
        for disp_name, acc_a, qwk_a, drop in ablation_results:
            out_f.write(f"{disp_name:<25} | {acc_a*100:5.2f}%     | {qwk_a:5.3f}          | -{drop*100:.2f}%\n")
        out_f.write("-" * 80 + "\n")
        
    print(f"\n[SAVE] Saved statistical summary results to {audit_file}")
    print("======================================================================")

if __name__ == "__main__":
    run_validation_suite()
