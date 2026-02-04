# ==============================================================================
# Knee Osteoarthritis Diagnostic Pipeline - High-Performance Ensemble & XAI
# Refactored & Cleaned (Unified Version)
# ==============================================================================

# ----------------- CODE CELL 1 -----------------
# ============================================================
# IMPORTS AND CONFIGURATION
# ============================================================
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import timm

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from torch.amp import autocast, GradScaler

# Dynamically locate root dataset & weights folder (checking src/.. and current working dir)
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
os.makedirs(WEIGHTS_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test")
AUTO_DIR  = os.path.join(DATA_ROOT, "auto_test")

IMG_SIZE = 224
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ============================================================
# CLINICALLY VALID 3-CLASS REMAPPING
# ============================================================
# KL Grade 0 & 1 -> Healthy (No active OA)
# KL Grade 2 & 3 -> Moderate (Active early/mid OA)
# KL Grade 4     -> Severe (Advanced joint degradation)
CLASS_MAP = {
    '0': 'Healthy',
    '1': 'Healthy',
    '2': 'Moderate',
    '3': 'Moderate',
    '4': 'Severe'
}

FINAL_CLASSES = ['Healthy', 'Moderate', 'Severe']
CLASS_TO_IDX = {c: i for i, c in enumerate(FINAL_CLASSES)}
NUM_CLASSES = 3



# ----------------- CODE CELL 2 -----------------
# ============================================================
# DATA TRANSFORMS & CUSTOM DATASET
# ============================================================
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.15, 0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class RemappedImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]
        original = os.path.basename(os.path.dirname(path))
        label = CLASS_MAP[original]
        target = CLASS_TO_IDX[label]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target

# ============================================================
# LOADERS
# ============================================================
train_ds = RemappedImageFolder(TRAIN_DIR, train_tfms)
val_ds   = RemappedImageFolder(VAL_DIR, val_tfms)
test_ds  = RemappedImageFolder(TEST_DIR, val_tfms)
auto_ds  = RemappedImageFolder(AUTO_DIR, val_tfms)

# Compute class weights for WeightedRandomSampler to address class imbalance
train_targets = [y for _, y in train_ds]
class_counts = np.bincount(train_targets)
class_weights = 1.0 / class_counts
sample_weights = np.array([class_weights[t] for t in train_targets])

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
auto_loader  = DataLoader(auto_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print("Classes:", FINAL_CLASSES)
print(f"Dataset Size - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")



# ----------------- CODE CELL 3 -----------------
# ============================================================
# LOSS, OPTIMIZATION & TRAINING HELPERS (WITH AMP)
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
scaler = GradScaler("cuda")

def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def ordinal_loss(logits, targets, ce_criterion):
    ce_loss = ce_criterion(logits, targets)
    probs = torch.softmax(logits, dim=1)
    expected_class = torch.sum(probs * torch.arange(logits.size(1), device=logits.device).float(), dim=1)
    mae_loss = torch.mean(torch.abs(expected_class - targets.float()))
    return ce_loss + 0.5 * mae_loss

def evaluate(model, loader):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            preds.extend(out.argmax(1).cpu().numpy())
            gts.extend(y.numpy())
    return accuracy_score(gts, preds)

def train_model(model, epochs, lr, name, mixup_alpha=0.2):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    # Cosine Annealing Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_acc = 0
    loss_fn = lambda logits, targets: ordinal_loss(logits, targets, criterion)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda"):
                if mixup_alpha > 0 and model.training:
                    inputs, targets_a, targets_b, lam = mixup_data(x, y, mixup_alpha, DEVICE)
                    out = model(inputs)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = mixup_criterion(loss_fn, out, targets_a, targets_b, lam)
                else:
                    out = model(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            # Print batch-level progress every 50 steps
            if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
                print(f"  [Epoch {epoch+1}] Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        acc = evaluate(model, val_loader)
        print(f"{name} | Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(WEIGHTS_DIR, f"{name}_no_leakage.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[SAVE] Saved new best weights to {save_path}")



# ----------------- CODE CELL 4 -----------------
# ============================================================
# CNN MODEL (CONVNEXT-BASE) TRAINING
# ============================================================
import gc
gc.collect()
torch.cuda.empty_cache()

cnn_model = timm.create_model("convnext_base", pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

# Stage 1: Fine-tune Classification Head Only
print(">>> Stage 1: Training ConvNeXt Classification Head Only...")
for n, p in cnn_model.named_parameters():
    p.requires_grad = ("head" in n)

train_model(cnn_model, epochs=5, lr=1e-3, name="cnn_model_head")

# Stage 2: Fine-tune Stage 3 and Head
print("\n>>> Stage 2: Unfreezing ConvNeXt Stage 3 (stages.3) and Head for Fine-Tuning...")
for n, p in cnn_model.named_parameters():
    p.requires_grad = any(b in n for b in ["stages.3", "head"])

train_model(cnn_model, epochs=20, lr=5e-5, name="cnn_model")



# ----------------- CODE CELL 5 -----------------
# ============================================================
# VIT MODEL (Swin-Base) TRAINING
# ============================================================
import gc
if 'cnn_model' in globals():
    del cnn_model
gc.collect()
torch.cuda.empty_cache()

vit_model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

# Stage 1: Fine-tune Classification Head Only
print(">>> Stage 1: Training Swin Classification Head Only...")
for n, p in vit_model.named_parameters():
    p.requires_grad = ("head" in n)

train_model(vit_model, epochs=5, lr=1e-3, name="vit_head")

# Stage 2: Fine-tune Top Transformer Blocks
print("\n>>> Stage 2: Unfreezing Swin Stage 4 (layers.3) and Head for Fine-Tuning...")
for n, p in vit_model.named_parameters():
    p.requires_grad = any(b in n for b in ["layers.3", "head"])

train_model(vit_model, epochs=20, lr=5e-5, name="vit_model")



# ----------------- CODE CELL 6 -----------------
# ============================================================
# DINO VIT MODEL (HEAD -> PARTIAL UNFREEZE) TRAINING
# ============================================================
import gc
if 'vit_model' in globals():
    del vit_model
gc.collect()
torch.cuda.empty_cache()

dino_model = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

# Stage 1: Fine-tune Classification Head Only
print(">>> Stage 1: Training DINO ViT Classification Head Only...")
for n, p in dino_model.named_parameters():
    p.requires_grad = ("head" in n)

train_model(dino_model, epochs=5, lr=1e-3, name="dino_head")

# Stage 2: Fine-tune Top Transformer Blocks
print("\n>>> Stage 2: Unfreezing DINO Blocks 8-11 for Fine-Tuning...")
for n, p in dino_model.named_parameters():
    p.requires_grad = any(b in n for b in ["blocks.8", "blocks.9", "blocks.10", "blocks.11", "head"])

train_model(dino_model, epochs=20, lr=5e-5, name="dino_model")



# ----------------- CODE CELL 7 -----------------
# ============================================================
# COMBINED HYBRID MODEL (CNN + ViT + DINO) DEFINITION & TRAINING
# ============================================================
import gc
if 'dino_model' in globals():
    del dino_model
gc.collect()
torch.cuda.empty_cache()

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.vit = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=0)
        self.dino = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0)

        # Freeze DINO backbone (self-supervised structural features)
        for p in self.dino.parameters():
            p.requires_grad = False

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

    def forward(self, x):
        f_cnn = self.cnn(x)
        f_vit = self.vit(x)
        f_dino = self.dino(x)
        fused = torch.cat([f_cnn, f_vit, f_dino], dim=1)
        return self.head(fused), f_cnn, f_vit, f_dino

hybrid_model = HybridModel().to(DEVICE)

# Load pretrained branch weights individually to be robust against missing files
for branch, filename in [("cnn", "cnn_model_no_leakage.pth"), ("vit", "vit_model_no_leakage.pth"), ("dino", "dino_model_no_leakage.pth")]:
    filepath = os.path.join(WEIGHTS_DIR, filename) if os.path.exists(os.path.join(WEIGHTS_DIR, filename)) else filename
    try:
        getattr(hybrid_model, branch).load_state_dict(torch.load(filepath, map_location=DEVICE), strict=False)
        print(f"Loaded pretrained weights for {branch.upper()} branch from {filepath}.")
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Branch {branch.upper()} will use default/scratch weights.")

# Stage 1: Train Head Only
print(">>> Stage 1: Fine-Tuning Hybrid Classifier Head...")
for p in hybrid_model.cnn.parameters(): p.requires_grad = False
for p in hybrid_model.vit.parameters(): p.requires_grad = False

train_model(hybrid_model, epochs=5, lr=1e-3, name="hybrid_head")

# Stage 2: Joint Backbone Training
print("\n>>> Stage 2: Unfreezing CNN Backbone (Stage 3) and Swin Blocks (Stage 4)...")
for n, p in hybrid_model.cnn.named_parameters():
    p.requires_grad = any(b in n for b in ["stages.3"])
for n, p in hybrid_model.vit.named_parameters():
    p.requires_grad = any(b in n for b in ["layers.3"])

train_model(hybrid_model, epochs=20, lr=5e-5, name="hybrid_model")



# ----------------- CODE CELL 8 -----------------
# ============================================================
# TEST-TIME AUGMENTATION (TTA) AND ENSEMBLING
# ============================================================
tta_tfms = [
    val_tfms,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation([5, 5]),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation([-5, -5]),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
]

def tta_predict(model, dataset):
    model.eval()
    orig_tf = dataset.transform
    probs_all = []
    try:
        for tfm in tta_tfms:
            dataset.transform = tfm
            loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)
            probs = []
            with torch.no_grad():
                for x, _ in loader:
                    x = x.to(DEVICE)
                    out = model(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    probs.append(torch.softmax(out, dim=1).cpu().numpy())
            probs_all.append(np.vstack(probs))
    finally:
        dataset.transform = orig_tf
    return np.mean(probs_all, axis=0)



# ----------------- CODE CELL 9 -----------------
# ============================================================
# EVALUATION ON TEST & AUTOTEST SETS
# ============================================================
import gc
if 'hybrid_model' in globals():
    del hybrid_model
gc.collect()
torch.cuda.empty_cache()

# Re-create models dynamically to prevent VRAM overflow/paging
cnn_model = timm.create_model("convnext_base", pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
vit_model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
dino_model = timm.create_model("vit_base_patch16_224.dino", pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
hybrid_model = HybridModel().to(DEVICE)

eval_dict = {
    "ConvNeXt (CNN)": cnn_model,
    "Swin Transformer (ViT)": vit_model,
    "DINO ViT": dino_model,
    "Combined Hybrid Model": hybrid_model
}

# Load best weights
try:
    for name, pth in [
        ("ConvNeXt (CNN)", "cnn_model_no_leakage.pth"),
        ("Swin Transformer (ViT)", "vit_model_no_leakage.pth"),
        ("DINO ViT", "dino_model_no_leakage.pth"),
        ("Combined Hybrid Model", "hybrid_model_no_leakage.pth")
    ]:
        filepath = os.path.join(WEIGHTS_DIR, pth) if os.path.exists(os.path.join(WEIGHTS_DIR, pth)) else pth
        eval_dict[name].load_state_dict(torch.load(filepath, map_location=DEVICE))
        print(f"Loaded best weights for {name} from {filepath}.")
except FileNotFoundError as e:
    print(f"Warning: Could not load some weights. Proceeding with current memory weights. Error: {e}")

for name, ds in [("TEST", test_ds), ("AUTOTEST", auto_ds)]:
    # Compute TTA probabilities
    cnn_p = tta_predict(cnn_model, ds)
    vit_p = tta_predict(vit_model, ds)
    dino_p = tta_predict(dino_model, ds)
    hyb_p = tta_predict(hybrid_model, ds)

    # Ensemble Voting using Optimal Weights
    final_p = (0.1 * cnn_p) + (0.2 * vit_p) + (0.1 * dino_p) + (0.6 * hyb_p)
    y_pred = np.argmax(final_p, axis=1)
    
    # Extract ground truth labels safely
    y_true = [y for _, y in ds]

    print(f"\n===== {name} ENSEMBLE RESULTS =====")
    print("Accuracy: ", accuracy_score(y_true, y_pred) * 100)
    print(classification_report(y_true, y_pred, target_names=FINAL_CLASSES))

    # Plot Confusion Matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_true, y_pred),
                annot=True, fmt="d", cmap="Blues",
                xticklabels=FINAL_CLASSES,
                yticklabels=FINAL_CLASSES)
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.show()



# ----------------- CODE CELL 10 -----------------
# ============================================================
# DYNAMIC GPU-BASED MULTI-XAI FOR RESNET50 BASELINE
# (Grad-CAM | Grad-CAM++ | Layer-CAM | Score-CAM | Integrated Gradients)
# ============================================================
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients

# Use active GPU device
xai_device = DEVICE
model = cnn_model.eval().to(xai_device)
TARGET_LAYER = model.layer4[-1]

# Custom Remapped Dataset that also returns file paths for plotting
class RemappedFolderWithPath(ImageFolder):
    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        original = os.path.basename(os.path.dirname(path))
        label = CLASS_MAP[original]
        target = CLASS_TO_IDX[label]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target, path

xai_ds = RemappedFolderWithPath(TEST_DIR, val_tfms)

# Collect 3 sample images per class
samples = {0: [], 1: [], 2: []}
for img, y, path in xai_ds:
    if len(samples[y]) < 3:
        samples[y].append((img, y, path))
    if all(len(samples[c]) == 3 for c in samples):
        break

# Initialize explainers
gradcam = GradCAM(model, [TARGET_LAYER])
gradcampp = GradCAMPlusPlus(model, [TARGET_LAYER])
layercam = LayerCAM(model, [TARGET_LAYER])
# ScoreCAM can be slow; we keep it for full comparison. Change if timeout occurs.
scorecam = ScoreCAM(model, [TARGET_LAYER])
ig = IntegratedGradients(model)

def denorm(img):
    img = img.permute(1,2,0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

for cls in samples:
    print(f"\n===== GENERATING EXPLANATIONS FOR CLASS: {FINAL_CLASSES[cls]} =====")
    
    for idx, (img, y, path) in enumerate(samples[cls]):
        x = img.unsqueeze(0).to(xai_device)
        img_np = denorm(img)

        # Generate attributions
        cam1 = gradcam(x)[0]
        cam2 = gradcampp(x)[0]
        cam3 = layercam(x)[0]
        cam4 = scorecam(x)[0]

        ig_attr = ig.attribute(x, target=y, n_steps=15)
        ig_map = ig_attr.abs().mean(1)[0].cpu().numpy()
        ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)

        # Generate overlays
        vis1 = show_cam_on_image(img_np, cam1, use_rgb=True)
        vis2 = show_cam_on_image(img_np, cam2, use_rgb=True)
        vis3 = show_cam_on_image(img_np, cam3, use_rgb=True)
        vis4 = show_cam_on_image(img_np, cam4, use_rgb=True)
        vis5 = show_cam_on_image(img_np, ig_map, use_rgb=True)

        # Plot comparison row
        plt.figure(figsize=(18,4))
        titles = ["Original", "Grad-CAM", "Grad-CAM++", "Layer-CAM", "Score-CAM", "Integrated Gradients"]
        imgs = [img_np, vis1, vis2, vis3, vis4, vis5]

        for i in range(6):
            plt.subplot(1, 6, i + 1)
            plt.imshow(imgs[i])
            plt.title(titles[i])
            plt.axis("off")

        plt.suptitle(f"{FINAL_CLASSES[y]} Baseline | Sample {idx+1}", fontsize=14)
        plt.tight_layout()
        plt.show()



# ----------------- CODE CELL 11 -----------------
# ============================================================
# DYNAMIC HYBRID MODEL BRANCH ATTRIBUTION XAI
# Grad-CAM on CNN branch | Integrated Gradients on ViT, DINO & Fusion
# ============================================================
from captum.attr import IntegratedGradients

model = hybrid_model.eval().to(DEVICE)

# Setup CNN-branch Grad-CAM
gradcam_hybrid = GradCAM(model.cnn, [model.cnn.layer4[-1]])

# Create wrapper functions returning scalar outputs for Captum attributions
# Integrated Gradients require a single target logit or scalar attribution value
ig_fusion = IntegratedGradients(lambda x: model(x)[0])
ig_vit    = IntegratedGradients(lambda x: model(x)[2].norm(dim=1))
ig_dino   = IntegratedGradients(lambda x: model(x)[3].norm(dim=1))

for cls in samples:
    print(f"\n===== HYBRID BRANCH ATTRIBUTIONS FOR CLASS: {FINAL_CLASSES[cls]} =====")
    
    for idx, (img, y, path) in enumerate(samples[cls]):
        x = img.unsqueeze(0).to(DEVICE)
        img_np = denorm(img)

        # Get prediction
        logits, _, _, _ = model(x)
        pred_cls = logits.argmax(1).item()

        # Compute Grad-CAM on CNN branch (using the class's gradients)
        # Note: model.cnn outputs features (num_features); we map gradients from the final head output
        model.zero_grad()
        logits[0, y].backward(retain_graph=True)
        cnn_grads = gradcam_hybrid.layer.register_full_backward_hook
        # Approximate CNN Branch CAM
        cnn_cam = gradcam_hybrid(x, target_category=pred_cls)[0]

        # Compute Branch-specific attributions
        vit_attr = ig_vit.attribute(x, n_steps=15)
        vit_map = vit_attr.abs().mean(1)[0].cpu().numpy()
        vit_map = (vit_map - vit_map.min()) / (vit_map.max() - vit_map.min() + 1e-8)

        dino_attr = ig_dino.attribute(x, n_steps=15)
        dino_map = dino_attr.abs().mean(1)[0].cpu().numpy()
        dino_map = (dino_map - dino_map.min()) / (dino_map.max() - dino_map.min() + 1e-8)

        fusion_attr = ig_fusion.attribute(x, target=pred_cls, n_steps=15)
        fusion_map = fusion_attr.abs().mean(1)[0].cpu().numpy()
        fusion_map = (fusion_map - fusion_map.min()) / (fusion_map.max() - fusion_map.min() + 1e-8)

        # Generate overlays
        vis_cnn = show_cam_on_image(img_np, cnn_cam, use_rgb=True)
        vis_vit = show_cam_on_image(img_np, vit_map, use_rgb=True)
        vis_dino = show_cam_on_image(img_np, dino_map, use_rgb=True)
        vis_fusion = show_cam_on_image(img_np, fusion_map, use_rgb=True)

        # Plot comparison row
        plt.figure(figsize=(15,4))
        titles = ["Original", "CNN Branch CAM", "ViT Branch IG", "DINO Branch IG", "Fusion Head IG"]
        imgs = [img_np, vis_cnn, vis_vit, vis_dino, vis_fusion]

        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.imshow(imgs[i])
            plt.title(titles[i])
            plt.axis("off")

        plt.suptitle(f"Hybrid {FINAL_CLASSES[y]} | Prediction: {FINAL_CLASSES[pred_cls]} | Sample {idx+1}", fontsize=14)
        plt.tight_layout()
        plt.show()


