import os
import io
import time
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Maximize multi-threaded CPU throughput on Intel i3-1215U (6 cores / 8 threads)
torch.set_num_threads(8)

# --- 1. MODEL ARCHITECTURE ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class TransformerBottleneck(nn.Module):
    def __init__(self, in_ch=128, embed_dim=256, num_layers=4, num_heads=8, patch_size=4, feat_size=56):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = feat_size // patch_size  # 14
        self.num_patches = self.grid_size * self.grid_size  # 196

        self.patch_proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            activation='gelu',
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, in_ch, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.patch_proj(x)
        feat = feat.flatten(2).transpose(1, 2) + self.pos_embed
        trans_out = self.transformer(feat)
        trans_out = trans_out.transpose(1, 2).reshape(B, -1, self.grid_size, self.grid_size)
        out = self.reconstruct(trans_out)
        return out

class HybridCNNTransformerUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 128)
        self.bottleneck = TransformerBottleneck(in_ch=128, embed_dim=256, num_layers=4, num_heads=8, patch_size=4, feat_size=56)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32 + 64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        b = self.bottleneck(e3)

        d1 = self.up1(b)
        d1 = torch.cat([d1, p1], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        logits = self.final_conv(d2)
        return logits

# --- 2. FAST IN-MEMORY DATASET ---
class FastRAMBrainDataset(Dataset):
    def __init__(self, image_arrays, mask_arrays, is_train=True):
        """
        image_arrays: np.ndarray of shape (N, 224, 224, 3), dtype=uint8
        mask_arrays: np.ndarray of shape (N, 224, 224), dtype=uint8
        """
        self.images = image_arrays
        self.masks = mask_arrays
        self.is_train = is_train
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx].copy()
        mask_np = self.masks[idx].copy()

        # Fast numpy augmentations
        if self.is_train:
            if random.random() > 0.5:
                img_np = np.fliplr(img_np)
                mask_np = np.fliplr(mask_np)

            if random.random() > 0.5:
                img_np = np.flipud(img_np)
                mask_np = np.flipud(mask_np)

            # Random 90 degree rotation
            k = random.randint(0, 3)
            if k > 0:
                img_np = np.rot90(img_np, k)
                mask_np = np.rot90(mask_np, k)

        # To tensor & normalize
        img_t = torch.from_numpy(img_np.copy()).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_np.copy()).unsqueeze(0).float() / 255.0

        img_t = (img_t - self.mean) / self.std
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t

# --- 3. GATHER ALL SAMPLES & PRE-CACHE TO RAM ---
def load_all_data_to_ram(image_size=(224, 224)):
    base_dir = Path(__file__).parent.parent
    sample_pairs = []

    # Download dataset from Kaggle if needed and get path
    try:
        import kagglehub
        print("[DATA] Downloading/Locating dataset from Kaggle...", flush=True)
        ds_path = Path(kagglehub.dataset_download("sharatlaha/brain-ai-data"))
        seg_dir = None
        notumor_dir = None
        for p in ds_path.rglob('*'):
            if p.is_dir() and p.name == 'Segmentation' and p.parent.name == 'Segmentation_Data':
                seg_dir = p
            if p.is_dir() and p.name == 'notumor' and p.parent.name == 'Training' and p.parent.parent.name == 'Classification_Data':
                notumor_dir = p
    except Exception as e:
        print(f"[WARN] Failed to use kagglehub: {e}")
        ds_path = base_dir / "DATA_Cleaned"
        seg_dir = ds_path / "Segmentation_Data" / "Segmentation"
        notumor_dir = ds_path / "Classification_Data" / "Training" / "notumor"

    if seg_dir is None or not seg_dir.exists():
        seg_dir = base_dir / "DATA_Cleaned" / "Segmentation_Data" / "Segmentation"
    if notumor_dir is None or not notumor_dir.exists():
        notumor_dir = base_dir / "DATA_Cleaned" / "Classification_Data" / "Training" / "notumor"

    # 1. Positive segmentation dataset (2,192 scans with true masks)
    for cat in ['Glioma', 'Meningioma', 'Pituitary tumor']:
        cat_p = seg_dir / cat
        if cat_p.exists():
            scans = [p for p in cat_p.glob("enh_*.png") if not p.name.endswith("_mask.png")]
            for img_p in scans:
                mask_p = img_p.with_name(img_p.stem + "_mask.png")
                if mask_p.exists():
                    sample_pairs.append((str(img_p), str(mask_p)))

    pos_count = len(sample_pairs)

    # 2. Negative healthy control dataset (from Datasets/Training/notumor)
    neg_count = 0
    if notumor_dir.exists():
        notumor_files = sorted(list(notumor_dir.glob("*.jpg")))
        for p in notumor_files[:400]:
            sample_pairs.append((str(p), None))
            neg_count += 1

    total = len(sample_pairs)
    print(f"[DATA] Found {total} total scans: {pos_count} tumor pairs + {neg_count} healthy controls.", flush=True)
    print(f"[CACHE] Pre-caching all {total} scans into RAM at {image_size}...", flush=True)

    t0 = time.time()
    images_arr = np.zeros((total, image_size[0], image_size[1], 3), dtype=np.uint8)
    masks_arr = np.zeros((total, image_size[0], image_size[1]), dtype=np.uint8)

    for i, (img_path, mask_path) in enumerate(sample_pairs):
        try:
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB").resize(image_size, Image.BILINEAR)
                images_arr[i] = np.asarray(im_rgb)

            if mask_path is not None and os.path.exists(mask_path):
                with Image.open(mask_path) as mk:
                    mk_l = mk.convert("L").resize(image_size, Image.NEAREST)
                    masks_arr[i] = (np.asarray(mk_l) > 127).astype(np.uint8) * 255
            else:
                # Blank zero mask for healthy control
                masks_arr[i] = 0
        except Exception as e:
            print(f"[WARN] Error reading {img_path}: {e}", flush=True)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  Cached {i + 1}/{total} samples ({(i + 1)/total*100:.1f}%) in {time.time() - t0:.1f}s", flush=True)

    ram_mb = (images_arr.nbytes + masks_arr.nbytes) / (1024 * 1024)
    print(f"[CACHE] Done! Total RAM cached: {ram_mb:.1f} MB in {time.time() - t0:.1f}s", flush=True)
    return images_arr, masks_arr

# --- 4. LOSS FUNCTION & METRICS ---
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return bce + dice

def calculate_dice(logits, targets, threshold=0.5, smooth=1e-5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    p_flat = preds.view(-1)
    t_flat = targets.view(-1)
    intersection = (p_flat * t_flat).sum().item()
    total = p_flat.sum().item() + t_flat.sum().item()
    if total == 0:
        return 1.0
    return (2.0 * intersection + smooth) / (total + smooth)

# --- 5. TRAINING ROUTINE ---
def run_training(num_epochs=10, batch_size=16, lr=3e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Training on: {device} with {torch.get_num_threads()} CPU threads", flush=True)

    images_all, masks_all = load_all_data_to_ram(image_size=(224, 224))
    total_samples = len(images_all)

    # Deterministic train/validation split (85% / 15%)
    indices = list(range(total_samples))
    random.seed(42)
    random.shuffle(indices)

    split = int(0.85 * total_samples)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = FastRAMBrainDataset(images_all[train_idx], masks_all[train_idx], is_train=True)
    val_ds = FastRAMBrainDataset(images_all[val_idx], masks_all[val_idx], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = HybridCNNTransformerUNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = DiceBCELoss()

    save_dir = Path(__file__).parent.parent / "Main Files"
    save_path = save_dir / "best_hybrid_transunet.pth"
    frontend_save_path = Path(__file__).parent / "best_hybrid_transunet.pth"

    best_val_dice = 0.0

    if save_path.exists():
        print(f"[RESUME] Found existing checkpoint! Loading weights from {save_path}...", flush=True)
        model.load_state_dict(torch.load(str(save_path), map_location=device, weights_only=True))

    print(f"[START] Training {num_epochs} epochs | Batch size {batch_size} | LR {lr}", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for b_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            dice = calculate_dice(logits, masks)
            train_loss += loss.item() * imgs.size(0)
            train_dice += dice * imgs.size(0)

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch:02d} | Batch {b_idx+1:03d}/{len(train_loader)} | Batch Loss: {loss.item():.4f} | Batch Dice: {dice:.4f}", flush=True)

        scheduler.step()
        train_loss /= len(train_ds)
        train_dice /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                dice = calculate_dice(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                val_dice += dice * imgs.size(0)

        val_loss /= len(val_ds)
        val_dice /= len(val_ds)
        duration = time.time() - epoch_start

        print(f"--> Epoch [{epoch:02d}/{num_epochs:02d}] Finished in {duration:.1f}s | "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}", flush=True)

        # Progressive checkpointing - save whenever validation score improves
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), str(save_path))
            torch.save(model.state_dict(), str(frontend_save_path))
            print(f"  [SAVED] New best model saved (Val Dice: {best_val_dice:.4f})", flush=True)

    print("=" * 70, flush=True)
    print(f"[COMPLETE] Final Best Validation Dice Score: {best_val_dice:.4f}", flush=True)

if __name__ == '__main__':
    run_training(num_epochs=10, batch_size=16, lr=3e-4)
