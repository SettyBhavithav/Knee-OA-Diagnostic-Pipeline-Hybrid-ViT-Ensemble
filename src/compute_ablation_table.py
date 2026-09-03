import os
import sys
import torch
import torch.nn as nn
import numpy as np
import timm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score

SCRIPT_DIR = r"C:\Users\setty\OneDrive\Desktop\deeplearning koa"
DATA_ROOT = os.path.join(SCRIPT_DIR, "dataset")
WEIGHTS_DIR = SCRIPT_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
IMG_SIZE = 384
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
    def forward(self, x, ablate=None):
        f_cnn = self.cnn(x)
        f_vit = self.vit(x)
        f_dino = self.dino(x)
        if ablate == 'cnn': f_cnn = torch.zeros_like(f_cnn)
        elif ablate == 'vit': f_vit = torch.zeros_like(f_vit)
        elif ablate == 'dino': f_dino = torch.zeros_like(f_dino)
        fused = torch.cat([f_cnn, f_vit, f_dino], dim=1)
        return self.head(fused)

def safe_load_state_dict(model, path, device):
    state_dict = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    final_state = {k: v for k, v in clean_state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(final_state, strict=False)
    model.to(device)

def evaluate_ablation(hybrid, test_loader, ablate_mode=None):
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in test_loader:
            x = x.to(DEVICE, non_blocking=True)
            y_true.extend(y.numpy())
            
            # Single pass without TTA to match the 83.44% baseline in Table IX
            out = hybrid(x, ablate=ablate_mode)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_pred.extend(preds)
            
    acc = accuracy_score(y_true, y_pred) * 100
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return acc, qwk

def main():
    test_images, test_labels = get_eval_images(DATA_ROOT)
    test_ds = FastPatientDataset(test_images, test_labels, val_tfms)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if DEVICE=='cuda' else False)
    
    hybrid = HybridModel()
    safe_load_state_dict(hybrid, os.path.join(WEIGHTS_DIR, "hybrid_model_no_leakage.pth"), DEVICE)
    hybrid.eval()
    
    acc_none, qwk_none = evaluate_ablation(hybrid, test_loader, None)
    acc_cnn, qwk_cnn = evaluate_ablation(hybrid, test_loader, 'cnn')
    acc_vit, qwk_vit = evaluate_ablation(hybrid, test_loader, 'vit')
    acc_dino, qwk_dino = evaluate_ablation(hybrid, test_loader, 'dino')
    
    print("=" * 60)
    print(f"TABLE IX: Branch Ablation Study Results")
    print("=" * 60)
    print(f"{'Ablated Branch':<25} | {'Accuracy (%)':<15} | {'QWK':<10}")
    print("-" * 60)
    print(f"{'None (Full Hybrid)':<25} | {acc_none:15.2f} | {qwk_none:10.3f}")
    print(f"{'ConvNeXt (Local)':<25} | {acc_cnn:15.2f} | {qwk_cnn:10.3f}")
    print(f"{'Swin ViT (Global)':<25} | {acc_vit:15.2f} | {qwk_vit:10.3f}")
    print(f"{'DINO ViT (Structural)':<25} | {acc_dino:15.2f} | {qwk_dino:10.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
