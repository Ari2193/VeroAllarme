"""
Quick training script for demo purposes
Trains on a small subset for fast evaluation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
from datetime import datetime

from core.relevance_mask import RelevanceMaskModel
from core.motion_detection import MotionDetector
from datasets.motion_dataset import ContrastiveMotionDataset, SegmentationDataset

# NTXent Loss (copied from train_relevance_model.py)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negative):
        # Normalize embeddings
        anchor = nn.functional.normalize(anchor, dim=1)
        positive = nn.functional.normalize(positive, dim=1)
        negative = nn.functional.normalize(negative, dim=1)
        
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature
        
        # Contrastive loss
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        loss = nn.functional.cross_entropy(logits, labels)
        return loss

print("=" * 80)
print("🚀 VeroAllarme - Quick Training Demo")
print("=" * 80)

# Config
DATA_DIR = "../data/training/camera-events/Factory"
OUTPUT_DIR = "model_checkpoints"
BATCH_SIZE = 4
CONTRASTIVE_EPOCHS = 5
SEGMENTATION_EPOCHS = 5
MAX_SEQUENCES = 50  # Use only 50 sequences for speed
DEVICE = "cpu"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

print(f"\nConfiguration:")
print(f"  Data: {DATA_DIR}")
print(f"  Max sequences: {MAX_SEQUENCES}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Contrastive epochs: {CONTRASTIVE_EPOCHS}")
print(f"  Segmentation epochs: {SEGMENTATION_EPOCHS}")
print(f"  Device: {DEVICE}")

# ==================== Phase 1: Contrastive Learning ====================

print("\n" + "=" * 80)
print("PHASE 1: Contrastive Learning")
print("=" * 80)

print("\nLoading dataset...")
full_dataset = ContrastiveMotionDataset(
    DATA_DIR,
    motion_detector=MotionDetector(),
    cache_detections=True
)

# Use subset
if len(full_dataset) > MAX_SEQUENCES:
    indices = list(range(min(MAX_SEQUENCES, len(full_dataset))))
    dataset = Subset(full_dataset, indices)
else:
    dataset = full_dataset

print(f"✓ Dataset size: {len(dataset)} sequences")

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Avoid multiprocessing issues
    drop_last=True
)

print(f"✓ Batches per epoch: {len(dataloader)}")

# Model
print("\nInitializing model...")
model = RelevanceMaskModel(
    feature_dim=128,
    mode="contrastive",
    pretrained=False  # Skip pretrained for speed
).to(DEVICE)

criterion = NTXentLoss(temperature=0.07)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("✓ Model initialized")

# Training
print(f"\nTraining for {CONTRASTIVE_EPOCHS} epochs...")
contrastive_losses = []

for epoch in range(1, CONTRASTIVE_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{CONTRASTIVE_EPOCHS}")
    
    for anchor, positive, negative in pbar:
        anchor = anchor.to(DEVICE)
        positive = positive.to(DEVICE)
        negative = negative.to(DEVICE)
        
        # Forward
        anchor_proj = model(anchor)
        positive_proj = model(positive)
        negative_proj = model(negative)
        
        loss = criterion(anchor_proj, positive_proj, negative_proj)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = epoch_loss / len(dataloader)
    contrastive_losses.append(avg_loss)
    
    print(f"\nEpoch {epoch}: Loss = {avg_loss:.4f}")

# Save contrastive model
contrastive_path = Path(OUTPUT_DIR) / "quick_contrastive.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": CONTRASTIVE_EPOCHS,
    "loss": contrastive_losses[-1]
}, contrastive_path)

print(f"\n✓ Saved contrastive model: {contrastive_path}")

# ==================== Phase 2: Segmentation ====================

print("\n" + "=" * 80)
print("PHASE 2: Segmentation Fine-tuning")
print("=" * 80)

print("\nLoading segmentation dataset...")
seg_full_dataset = SegmentationDataset(
    DATA_DIR,
    motion_detector=MotionDetector(),
    target_size=(224, 224)
)

# Use subset
if len(seg_full_dataset) > MAX_SEQUENCES:
    seg_indices = list(range(min(MAX_SEQUENCES, len(seg_full_dataset))))
    seg_dataset = Subset(seg_full_dataset, seg_indices)
else:
    seg_dataset = seg_full_dataset

print(f"✓ Dataset size: {len(seg_dataset)} samples")

# Split train/val
train_size = int(0.8 * len(seg_dataset))
val_size = len(seg_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    seg_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Create segmentation model with pre-trained encoder
seg_model = RelevanceMaskModel(
    feature_dim=128,
    mode="segmentation",
    pretrained=False
)

# Load contrastive encoder weights
seg_model.encoder.load_state_dict(model.encoder.state_dict())
seg_model = seg_model.to(DEVICE)

seg_criterion = nn.CrossEntropyLoss()
seg_optimizer = optim.Adam(seg_model.parameters(), lr=0.0001)

print("✓ Segmentation model initialized with pre-trained encoder")

# Training
print(f"\nTraining for {SEGMENTATION_EPOCHS} epochs...")
seg_train_losses = []
seg_val_losses = []
seg_val_ious = []

for epoch in range(1, SEGMENTATION_EPOCHS + 1):
    # Train
    seg_model.train()
    train_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{SEGMENTATION_EPOCHS} [Train]")
    
    for images, masks in pbar:
        images = images.to(DEVICE)
        
        # Debug: check mask shape
        # print(f"DEBUG: masks.shape = {masks.shape}")
        
        # Masks might be (B, 1, H, W) or (B, H, W)
        if len(masks.shape) == 3:
            # (B, H, W) -> add channel dimension
            masks = masks.unsqueeze(1)
        
        # Now masks are (B, 1, H, W), resize to (B, 1, 32, 32)
        masks = nn.functional.interpolate(
            masks.float(), 
            size=(32, 32), 
            mode='nearest'
        ).squeeze(1).long().to(DEVICE)
        
        logits = seg_model(images)
        loss = seg_criterion(logits, masks)
        
        seg_optimizer.zero_grad()
        loss.backward()
        seg_optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_train_loss = train_loss / len(train_loader)
    seg_train_losses.append(avg_train_loss)
    
    # Validate
    seg_model.eval()
    val_loss = 0.0
    val_iou = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            
            # Handle mask dimensions
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            # Resize to model output size
            masks = nn.functional.interpolate(
                masks.float(), 
                size=(32, 32), 
                mode='nearest'
            ).squeeze(1).long().to(DEVICE)
            
            logits = seg_model(images)
            loss = seg_criterion(logits, masks)
            
            preds = torch.argmax(logits, dim=1)
            intersection = ((preds == 1) & (masks == 1)).sum().item()
            union = ((preds == 1) | (masks == 1)).sum().item()
            iou = intersection / (union + 1e-6)
            
            val_loss += loss.item()
            val_iou += iou
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    
    seg_val_losses.append(avg_val_loss)
    seg_val_ious.append(avg_val_iou)
    
    print(f"\nEpoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
          f"Val Loss = {avg_val_loss:.4f}, Val IoU = {avg_val_iou:.4f}")

# Save segmentation model
seg_path = Path(OUTPUT_DIR) / "quick_segmentation.pth"
torch.save({
    "model_state_dict": seg_model.state_dict(),
    "optimizer_state_dict": seg_optimizer.state_dict(),
    "epoch": SEGMENTATION_EPOCHS,
    "loss": seg_val_losses[-1],
    "iou": seg_val_ious[-1]
}, seg_path)

print(f"\n✓ Saved segmentation model: {seg_path}")

# Create complete model
complete_model = RelevanceMaskModel(mode="both", pretrained=False)
complete_model.encoder.load_state_dict(seg_model.encoder.state_dict())
complete_model.segmentation_head.load_state_dict(seg_model.segmentation_head.state_dict())

complete_path = Path(OUTPUT_DIR) / "quick_complete.pth"
torch.save({
    "model_state_dict": complete_model.state_dict()
}, complete_path)

print(f"✓ Saved complete model: {complete_path}")

# ==================== Summary ====================

print("\n" + "=" * 80)
print("📊 TRAINING SUMMARY")
print("=" * 80)

print(f"\nPhase 1 - Contrastive Learning:")
print(f"  Initial Loss: {contrastive_losses[0]:.4f}")
print(f"  Final Loss: {contrastive_losses[-1]:.4f}")
print(f"  Improvement: {((contrastive_losses[0] - contrastive_losses[-1]) / contrastive_losses[0] * 100):.1f}%")

print(f"\nPhase 2 - Segmentation:")
print(f"  Initial Val Loss: {seg_val_losses[0]:.4f}")
print(f"  Final Val Loss: {seg_val_losses[-1]:.4f}")
print(f"  Initial Val IoU: {seg_val_ious[0]:.4f}")
print(f"  Final Val IoU: {seg_val_ious[-1]:.4f}")
print(f"  IoU Improvement: {((seg_val_ious[-1] - seg_val_ious[0]) / (seg_val_ious[0] + 1e-6) * 100):.1f}%")

# Save training log
log_data = {
    "timestamp": datetime.now().isoformat(),
    "config": {
        "max_sequences": MAX_SEQUENCES,
        "batch_size": BATCH_SIZE,
        "contrastive_epochs": CONTRASTIVE_EPOCHS,
        "segmentation_epochs": SEGMENTATION_EPOCHS
    },
    "phase1_contrastive": {
        "losses": contrastive_losses,
        "final_loss": contrastive_losses[-1]
    },
    "phase2_segmentation": {
        "train_losses": seg_train_losses,
        "val_losses": seg_val_losses,
        "val_ious": seg_val_ious,
        "final_iou": seg_val_ious[-1]
    },
    "models": {
        "contrastive": str(contrastive_path),
        "segmentation": str(seg_path),
        "complete": str(complete_path)
    }
}

log_path = Path(OUTPUT_DIR) / "quick_training_log.json"
with open(log_path, 'w') as f:
    json.dump(log_data, f, indent=2)

print(f"\n✓ Training log saved: {log_path}")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel saved to: {complete_path}")
print(f"\nTo use the trained model:")
print(f"  from core.relevance_mask import RelevanceMaskPredictor")
print(f"  predictor = RelevanceMaskPredictor(model_path='{complete_path}')")
print("=" * 80)
