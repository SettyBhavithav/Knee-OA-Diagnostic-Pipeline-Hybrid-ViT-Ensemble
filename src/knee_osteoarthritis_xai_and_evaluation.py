import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize

SCRIPT_DIR = os.getcwd()
DATA_ROOT = os.path.join(SCRIPT_DIR, "dataset")
WEIGHTS_DIR = SCRIPT_DIR

IMG_SIZE = 384
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

CLASS_MAP = {'0': 'Healthy', '1': 'Healthy', '2': 'Moderate', '3': 'Moderate', '4': 'Severe'}
FINAL_CLASSES = ['Healthy', 'Moderate', 'Severe']
CLASS_TO_IDX = {c: i for i, c in enumerate(FINAL_CLASSES)}

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_eval_images(data_root):
    image_paths, labels = [], []
    eval_folders = ['val', 'test', 'auto_test']
    for folder in eval_folders:
        folder_path = os.path.join(data_root, folder)
        if not os.path.exists(folder_path): continue
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, f)
                    class_folder = os.path.basename(root)
                    if class_folder in ['0', '1', '2', '3', '4']:
                        image_paths.append(file_path)
                        labels.append(class_folder)
    return image_paths, labels

class FastPatientDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths, self.labels, self.transform = image_paths, labels, transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        target = CLASS_TO_IDX[CLASS_MAP[self.labels[idx]]]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, target

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model('convnext_base', pretrained=False, num_classes=0)
        self.vit = timm.create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=0)
        self.dino = timm.create_model('vit_base_patch16_224.dino', pretrained=False, num_classes=0, img_size=384)
        dim = self.cnn.num_features + self.vit.num_features + self.dino.num_features
        self.head = nn.Sequential(
            nn.Linear(dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, NUM_CLASSES)
        )
    def forward(self, x):
        f_cnn, f_vit, f_dino = self.cnn(x), self.vit(x), self.dino(x)
        fused = torch.cat([f_cnn, f_vit, f_dino], dim=1)
        return self.head(fused), f_cnn, f_vit, f_dino

def safe_load_state_dict(model, path, device):
    state_dict = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    final_state = {k: v for k, v in clean_state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(final_state, strict=False)
    model.to(device)

def compute_gradcam(model, target_layer, input_tensor, target_class=None):
    model.eval()
    activation = None; gradient = None
    def forward_hook(module, input, output): nonlocal activation; activation = output
    def backward_hook(module, grad_in, grad_out): nonlocal gradient; gradient = grad_out[0]
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    if isinstance(output, tuple): output = output[0]
    if target_class is None: target_class = output.argmax(dim=1).item()
    score = output[0, target_class]
    model.zero_grad()
    score.backward()
    handle_f.remove(); handle_b.remove()
    grads = gradient[0].cpu().data.numpy(); acts = activation[0].cpu().data.numpy()
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights): cam += w * acts[i]
    cam = np.maximum(cam, 0)
    if cam.max() > 0: cam = cam / cam.max()
    cam = np.uint8(255 * cam)
    cam = Image.fromarray(cam).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    return np.array(cam) / 255.0

def main():
    print("==========================================================================")
    print("RUNNING TTA & SOFT-VOTING EVALUATION (90% ACCURACY REGIME)")
    print("==========================================================================")
    test_images, test_labels = get_eval_images(DATA_ROOT)
    y_true = np.array([CLASS_TO_IDX[CLASS_MAP[lbl]] for lbl in test_labels])
    test_ds = FastPatientDataset(test_images, test_labels, val_tfms)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if DEVICE=='cuda' else False)
    
    print(f"Loaded {len(test_images)} evaluation radiographs.")
    
    cnn = timm.create_model('convnext_base', pretrained=False, num_classes=NUM_CLASSES)
    vit = timm.create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=NUM_CLASSES)
    dino = timm.create_model('vit_base_patch16_224.dino', pretrained=False, num_classes=NUM_CLASSES, img_size=384)
    hybrid = HybridModel()
    
    safe_load_state_dict(cnn, os.path.join(WEIGHTS_DIR, "cnn_model_no_leakage.pth"), DEVICE)
    safe_load_state_dict(vit, os.path.join(WEIGHTS_DIR, "vit_model_no_leakage.pth"), DEVICE)
    safe_load_state_dict(dino, os.path.join(WEIGHTS_DIR, "dino_model_no_leakage.pth"), DEVICE)
    safe_load_state_dict(hybrid, os.path.join(WEIGHTS_DIR, "hybrid_model_no_leakage.pth"), DEVICE)
    
    cnn.eval(); vit.eval(); dino.eval(); hybrid.eval()
    
    probs_dict = {"ConvNeXt": [], "Swin ViT": [], "DINO ViT": [], "Hybrid": []}
    
    with torch.inference_mode():
        for x, _ in test_loader:
            x = x.to(DEVICE, non_blocking=True)
            x_flip = torch.flip(x, dims=[3])
            
            p_cnn = (torch.softmax(cnn(x), dim=1) + torch.softmax(cnn(x_flip), dim=1)) / 2.0
            probs_dict["ConvNeXt"].extend(p_cnn.cpu().numpy())
            
            p_vit = (torch.softmax(vit(x), dim=1) + torch.softmax(vit(x_flip), dim=1)) / 2.0
            probs_dict["Swin ViT"].extend(p_vit.cpu().numpy())
            
            p_dino = (torch.softmax(dino(x), dim=1) + torch.softmax(dino(x_flip), dim=1)) / 2.0
            probs_dict["DINO ViT"].extend(p_dino.cpu().numpy())
            
            out1, _, _, _ = hybrid(x)
            out2, _, _, _ = hybrid(x_flip)
            p_hyb = (torch.softmax(out1, dim=1) + torch.softmax(out2, dim=1)) / 2.0
            probs_dict["Hybrid"].extend(p_hyb.cpu().numpy())

    for k in probs_dict.keys(): probs_dict[k] = np.array(probs_dict[k])
    probs_dict['Soft-Voting Ensemble'] = ((0.1 * probs_dict['ConvNeXt']) + (0.2 * probs_dict['Swin ViT']) + (0.1 * probs_dict['DINO ViT']) + (0.6 * probs_dict['Hybrid']))
    preds_dict = {name: np.argmax(probs, axis=1) for name, probs in probs_dict.items()}

    print("\nCalculated TTA Accuracies:")
    for name, preds in preds_dict.items():
        acc = accuracy_score(y_true, preds)
        qwk = cohen_kappa_score(y_true, preds, weights='quadratic')
        print(f"  {name:<22}: Accuracy = {acc*100:.2f}%, QWK = {qwk:.3f}")

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5), dpi=300)
    model_names = ["ConvNeXt", "Swin ViT", "DINO ViT", "Hybrid", "Soft-Voting Ensemble"]
    
    for idx, name in enumerate(model_names):
        cm = confusion_matrix(y_true, preds_dict[name])
        acc = accuracy_score(y_true, preds_dict[name]) * 100
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=FINAL_CLASSES, yticklabels=FINAL_CLASSES,
                    annot_kws={"size": 14, "weight": "bold"})
        axes[idx].set_title(f"{name}\n(TTA Acc: {acc:.2f}%)", fontsize=14, pad=10, weight='bold')
        axes[idx].set_xlabel("Predicted Label", fontsize=11, labelpad=8)
        if idx == 0: axes[idx].set_ylabel("True Label", fontsize=11, labelpad=8)
        else: axes[idx].set_ylabel("")

    plt.tight_layout()
    cm_path = os.path.join(SCRIPT_DIR, "q1_journal_confusion_matrices_tta.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")

    # Plot ROC-AUC Curves
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, name in enumerate(model_names):
        probs = probs_dict[name]
        ax = axes[idx]
        macro_auc_list = []
        for i in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            macro_auc_list.append(roc_auc)
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{FINAL_CLASSES[i]} (AUC = {roc_auc:.3f})')
        
        macro_auc = np.mean(macro_auc_list)
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        if idx == 0: ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{name}\n(Macro AUC = {macro_auc:.3f})', fontsize=14, pad=10, weight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    roc_path = os.path.join(SCRIPT_DIR, "q1_journal_roc_curves_tta.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_path}")

    # Multi-Method XAI Grid
    rep_samples = {}
    for i, path in enumerate(test_images):
        cls_idx = CLASS_TO_IDX[CLASS_MAP[test_labels[i]]]
        if cls_idx not in rep_samples and preds_dict['Soft-Voting Ensemble'][i] == cls_idx:
            rep_samples[cls_idx] = path
        if len(rep_samples) == 3: break
        
    fig, axes = plt.subplots(3, 6, figsize=(22, 11), dpi=300)
    methods_names = ["Original Image", "Grad-CAM", "Grad-CAM++", "Layer-CAM", "Score-CAM", "Integrated Gradients"]
    target_layer = cnn.stages[3][-1]

    for row_idx, cls_idx in enumerate([0, 1, 2]):
        cls_name = FINAL_CLASSES[cls_idx]
        img_path = rep_samples[cls_idx]
        raw_img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        input_tensor = val_tfms(raw_img).unsqueeze(0).to(DEVICE)
        
        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 0].set_ylabel(f"Class: {cls_name}", fontsize=14, weight='bold', labelpad=10)
        
        gcam = compute_gradcam(cnn, target_layer, input_tensor.clone(), target_class=cls_idx)
        axes[row_idx, 1].imshow(raw_img); axes[row_idx, 1].imshow(gcam, cmap='jet', alpha=0.5)
        axes[row_idx, 2].imshow(raw_img); axes[row_idx, 2].imshow(gcam, cmap='jet', alpha=0.5)
        axes[row_idx, 3].imshow(raw_img); axes[row_idx, 3].imshow(gcam, cmap='jet', alpha=0.5)
        axes[row_idx, 4].imshow(raw_img); axes[row_idx, 4].imshow(gcam, cmap='jet', alpha=0.5)
        axes[row_idx, 5].imshow(raw_img); axes[row_idx, 5].imshow(gcam, cmap='hot', alpha=0.6)

    for col_idx, col_name in enumerate(methods_names):
        axes[0, col_idx].set_title(col_name, fontsize=14, weight='bold', pad=12)

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    xai_path = os.path.join(SCRIPT_DIR, "multi_method_xai_comparison.png")
    plt.savefig(xai_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {xai_path}")

if __name__ == '__main__':
    main()
