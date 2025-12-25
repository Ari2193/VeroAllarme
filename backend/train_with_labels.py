"""
Train location model with manual labels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import cv2

from core.location_relevance import LocationRelevanceNet

print("=" * 80)
print("🎯 Training with Manual Labels")
print("=" * 80)

class ManualLabelDataset(Dataset):
    """Dataset from YOLO format labels"""
    
    def __init__(self, label_file: str, samples_per_label: int = 1000):
        self.samples_per_label = samples_per_label
        self.boxes = []
        
        # Parse label file (YOLO format)
        label_path = Path(label_file)
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to x1, y1, x2, y2
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        self.boxes.append({
                            'class': class_id,
                            'x1': max(0, x1),
                            'y1': max(0, y1),
                            'x2': min(1, x2),
                            'y2': min(1, y2)
                        })
        
        print(f"✓ Loaded {len(self.boxes)} labeled boxes")
        if self.boxes:
            for i, box in enumerate(self.boxes):
                print(f"  Box {i}: class={box['class']}, "
                      f"x=[{box['x1']:.2f}-{box['x2']:.2f}], "
                      f"y=[{box['y1']:.2f}-{box['y2']:.2f}]")
    
    def __len__(self):
        return len(self.boxes) * self.samples_per_label
    
    def __getitem__(self, idx):
        box = self.boxes[idx // self.samples_per_label]
        
        # Random location
        x = np.random.random()
        y = np.random.random()
        
        # Check if inside box
        is_inside = (box['x1'] <= x <= box['x2'] and 
                     box['y1'] <= y <= box['y2'])
        
        # Label: inside box = relevant (1), outside = irrelevant (0)
        label = 1.0 if is_inside else 0.0
        
        coords = torch.tensor([x, y], dtype=torch.float32)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return coords, label_tensor

# Load dataset
label_file = "../data/1.txt"
dataset = ManualLabelDataset(label_file, samples_per_label=2000)

if len(dataset) == 0:
    print("❌ No labels found!")
    exit(1)

# Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

print(f"\n✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LocationRelevanceNet().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"✓ Model initialized (device: {device})")

# Training
epochs = 50
best_val_loss = float('inf')

print(f"\n{'='*80}")
print("Training...")
print(f"{'='*80}\n")

for epoch in range(1, epochs + 1):
    # Train
    model.train()
    train_loss = 0.0
    
    for coords, labels in train_loader:
        coords = coords.to(device)
        labels = labels.to(device)
        
        predictions = model(coords)
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for coords, labels in val_loader:
            coords = coords.to(device)
            labels = labels.to(device)
            
            predictions = model(coords)
            loss = criterion(predictions, labels)
            
            val_loss += loss.item()
            
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
    
    # Save best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_path = Path("model_checkpoints/Factory_location_best.pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "camera_name": "Factory",
            "epoch": epoch,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        }, model_path)

print(f"\n{'='*80}")
print("✅ Training Complete!")
print(f"   Best Val Loss: {best_val_loss:.4f}")
print(f"   Final Accuracy: {val_accuracy:.4f}")
print(f"{'='*80}")

# Test on grid
print("\n📊 Testing on grid...")
from core.location_relevance import LocationRelevancePredictor

predictor = LocationRelevancePredictor(camera_name="Factory")

# Test corners and center
test_points = [
    ("Top-left", 0.1, 0.1),
    ("Top-center", 0.5, 0.1),
    ("Top-right", 0.9, 0.1),
    ("Middle-left", 0.1, 0.5),
    ("Center", 0.5, 0.65),  # Should be relevant (inside box)
    ("Middle-right", 0.9, 0.5),
    ("Bottom-left", 0.1, 0.9),
    ("Bottom-center", 0.5, 0.9),
    ("Bottom-right", 0.9, 0.9),
]

print("\nLocation Tests (normalized 0-1):")
for name, x, y in test_points:
    is_rel, score = predictor.is_location_relevant(
        int(x * 704), int(y * 576), 704, 576
    )
    status = "✓ RELEVANT" if is_rel else "✗ NOT RELEVANT"
    print(f"  {name:15s} ({x:.1f}, {y:.1f}): {status:15s} (score: {score:.3f})")

# Generate visualization
print("\n🎨 Generating visualization...")

# Load sample frame
data_dir = Path("../data/training/camera-events/Factory")
event_dir = next(d for d in data_dir.iterdir() if d.is_dir())
images = sorted(event_dir.glob("*.jpg"))
frame = cv2.imread(str(images[0]))

visualization = predictor.visualize_heat_map(frame, alpha=0.6)

output_path = Path("data/relevance_outputs/location_heat_map_trained.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), visualization)

print(f"✅ Saved: {output_path}")
print("\n" + "="*80)
