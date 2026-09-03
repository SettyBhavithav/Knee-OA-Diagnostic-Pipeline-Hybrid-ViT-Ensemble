import os
import sys
import torch
import torch.nn as nn
import numpy as np
import timm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

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

def bootstrap_cis(y_true, y_pred, n_bootstraps=500):
    accs, f1s, kappas = [], [], []
    np.random.seed(42)
    for _ in range(n_bootstraps):
        while True:
            idx = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[idx])) == 3: break
        y_t_b, y_p_b = y_true[idx], y_pred[idx]
        accs.append(accuracy_score(y_t_b, y_p_b))
        f1s.append(f1_score(y_t_b, y_p_b, average='macro'))
        kappas.append(cohen_kappa_score(y_t_b, y_p_b, weights='quadratic'))
        
    def get_ci_str(vals, is_pct=True):
        sorted_vals = np.sort(vals)
        low = sorted_vals[int(0.025 * n_bootstraps)]
        high = sorted_vals[int(0.975 * n_bootstraps)]
        mean = np.mean(vals)
        if is_pct:
            return f"{mean*100:.2f}% [{low*100:.2f}%-{high*100:.2f}%]"
        else:
            return f"{mean:.3f} [{low:.3f}-{high:.3f}]"
            
    return get_ci_str(accs, True), get_ci_str(f1s, True), get_ci_str(kappas, False)

def main():
    test_images, test_labels = get_eval_images(DATA_ROOT)
    y_true = np.array([CLASS_TO_IDX[CLASS_MAP[lbl]] for lbl in test_labels])
    test_ds = FastPatientDataset(test_images, test_labels, val_tfms)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if DEVICE=='cuda' else False)
    
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
    probs_dict['Ensemble'] = ((0.1 * probs_dict['ConvNeXt']) + (0.2 * probs_dict['Swin ViT']) + (0.1 * probs_dict['DINO ViT']) + (0.6 * probs_dict['Hybrid']))
    preds_dict = {name: np.argmax(probs, axis=1) for name, probs in probs_dict.items()}

    print("=" * 110)
    print(f"{'Model':<12} | {'Accuracy [95% CI]':<32} | {'Macro-F1 [95% CI]':<32} | {'QWK [95% CI]':<25}")
    print("=" * 110)
    for name, preds in preds_dict.items():
        acc_str, f1_str, qwk_str = bootstrap_cis(y_true, preds, n_bootstraps=500)
        print(f"{name:<12} | {acc_str:<32} | {f1_str:<32} | {qwk_str:<25}")
    print("=" * 110)

if __name__ == "__main__":
    main()
