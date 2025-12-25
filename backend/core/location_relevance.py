"""
Location-Based Relevance Module

This module creates a spatial heat map that classifies every location in the frame
as relevant or irrelevant, independent of motion detection.

Usage:
    # Train on camera data
    trainer = LocationRelevanceTrainer(camera_name="Factory")
    trainer.train(epochs=10)
    
    # Use for inference
    predictor = LocationRelevancePredictor(camera_name="Factory")
    is_relevant = predictor.is_location_relevant(x, y)
    heat_map = predictor.get_heat_map()
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List
import json
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class LocationRelevanceConfig:
    """Configuration for location relevance model"""
    camera_name: str
    grid_size: Tuple[int, int] = (224, 224)  # Resolution of relevance map
    model_dir: str = "model_checkpoints"
    threshold: float = 0.5  # Relevance threshold
    

class LocationRelevanceNet(nn.Module):
    """
    Neural network that learns which locations in the frame are relevant.
    Takes (x, y) normalized coordinates and outputs relevance probability.
    """
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        
        # Coordinate embedding
        self.coord_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 2) normalized coordinates in [0, 1]
        Returns:
            relevance: (B, 1) relevance scores
        """
        x = self.coord_embed(coords)
        return self.network(x)


class LocationDataset(Dataset):
    """
    Dataset that extracts location labels from motion events.
    Locations with motion = relevant, others = irrelevant.
    """
    
    def __init__(
        self,
        data_dir: str,
        frame_size: Tuple[int, int] = (1920, 1080),
        samples_per_event: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.frame_w, self.frame_h = frame_size
        self.samples_per_event = samples_per_event
        
        # Load all motion events
        self.events = self._load_events()
        
        print(f"✓ Loaded {len(self.events)} events for location training")
    
    def _load_events(self) -> List[dict]:
        """Load motion event data"""
        from core.motion_detection import MotionDetector
        
        detector = MotionDetector()
        events = []
        
        event_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for event_dir in tqdm(event_dirs, desc="Loading events"):
            images = sorted(event_dir.glob("*.jpg"))
            if len(images) < 2:
                continue
            
            # Detect motion
            result = detector.detect_motion([str(p) for p in images[:5]])
            
            if not result.motion_detected:
                continue
            
            # Store event with motion regions
            events.append({
                "event_dir": event_dir,
                "motion_regions": result.motion_regions,
                "frame_h": result.frame_dimensions[0],
                "frame_w": result.frame_dimensions[1]
            })
        
        return events
    
    def __len__(self) -> int:
        return len(self.events) * self.samples_per_event
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a random location and its label (relevant/irrelevant)
        based on motion regions.
        """
        event_idx = idx // self.samples_per_event
        event = self.events[event_idx]
        
        # Random location in frame
        x = np.random.randint(0, self.frame_w)
        y = np.random.randint(0, self.frame_h)
        
        # Check if location is in any motion region
        is_relevant = False
        for region in event["motion_regions"]:
            x1, y1, x2, y2 = region.to_bbox()
            if x1 <= x <= x2 and y1 <= y <= y2:
                is_relevant = True
                break
        
        # Normalize coordinates to [0, 1]
        norm_x = x / self.frame_w
        norm_y = y / self.frame_h
        
        coords = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        label = torch.tensor([1.0 if is_relevant else 0.0], dtype=torch.float32)
        
        return coords, label


class LocationRelevanceTrainer:
    """Train location relevance model on camera data"""
    
    def __init__(
        self,
        camera_name: str,
        data_dir: Optional[str] = None,
        config: Optional[LocationRelevanceConfig] = None
    ):
        self.camera_name = camera_name
        self.config = config or LocationRelevanceConfig(camera_name=camera_name)
        
        if data_dir is None:
            data_dir = f"../data/training/camera-events/{camera_name}"
        
        self.data_dir = Path(data_dir)
        
        # Model
        self.model = LocationRelevanceNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"🎯 Location Relevance Trainer for {camera_name}")
        print(f"   Device: {self.device}")
    
    def train(
        self,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        samples_per_event: int = 100
    ):
        """Train the location relevance model"""
        
        print(f"\n{'='*80}")
        print(f"Training Location Relevance Model")
        print(f"{'='*80}")
        
        # Dataset
        dataset = LocationDataset(
            str(self.data_dir),
            samples_per_event=samples_per_event
        )
        
        # Split train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        # Training loop
        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
            for coords, labels in pbar:
                coords = coords.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                predictions = self.model(coords)
                loss = criterion(predictions, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for coords, labels in val_loader:
                    coords = coords.to(self.device)
                    labels = labels.to(self.device)
                    
                    predictions = self.model(coords)
                    loss = criterion(predictions, labels)
                    
                    val_loss += loss.item()
                    
                    # Accuracy
                    predicted_labels = (predictions > 0.5).float()
                    correct += (predicted_labels == labels).sum().item()
                    total += labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            print(f"\nEpoch {epoch}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, "
                  f"Val Accuracy = {val_accuracy:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(is_best=True)
            
            scheduler.step()
        
        # Save final model
        self.save_model(is_best=False)
        
        # Save training log
        log_data = {
            "camera_name": self.camera_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "best_val_loss": best_val_loss,
            "final_accuracy": val_accuracies[-1]
        }
        
        log_path = Path(self.config.model_dir) / f"{self.camera_name}_location_training_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ Training Complete!")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Final Accuracy: {val_accuracies[-1]:.4f}")
        print(f"{'='*80}")
    
    def save_model(self, is_best: bool = False):
        """Save model checkpoint"""
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            path = model_dir / f"{self.camera_name}_location_best.pth"
        else:
            path = model_dir / f"{self.camera_name}_location_final.pth"
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "camera_name": self.camera_name,
            "config": self.config
        }, path)


class LocationRelevancePredictor:
    """Use trained model to predict location relevance and generate heat maps"""
    
    def __init__(
        self,
        camera_name: str,
        model_path: Optional[str] = None,
        config: Optional[LocationRelevanceConfig] = None
    ):
        self.camera_name = camera_name
        self.config = config or LocationRelevanceConfig(camera_name=camera_name)
        
        # Load model
        self.model = LocationRelevanceNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path is None:
            model_path = f"{self.config.model_dir}/{camera_name}_location_best.pth"
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✓ Loaded location model from {model_path}")
        else:
            print(f"⚠ No trained model found at {model_path}, using random weights")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Cache for heat map
        self._heat_map_cache = None
        self._frame_size = None
    
    def is_location_relevant(
        self,
        x: int,
        y: int,
        frame_width: int = 1920,
        frame_height: int = 1080
    ) -> Tuple[bool, float]:
        """
        Check if a specific location is relevant.
        
        Args:
            x, y: Pixel coordinates
            frame_width, frame_height: Frame dimensions
        
        Returns:
            (is_relevant, confidence_score)
        """
        # Normalize coordinates
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        coords = torch.tensor([[norm_x, norm_y]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            score = self.model(coords).item()
        
        is_relevant = score > self.config.threshold
        
        return is_relevant, score
    
    def get_heat_map(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        resolution: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Generate full-frame heat map of relevance scores.
        
        Returns:
            heat_map: (H, W) array with values in [0, 1]
        """
        # Check cache
        if (self._heat_map_cache is not None and 
            self._frame_size == (frame_width, frame_height)):
            return self._heat_map_cache
        
        grid_h, grid_w = resolution
        heat_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        # Generate grid coordinates
        coords_list = []
        for i in range(grid_h):
            for j in range(grid_w):
                norm_x = j / grid_w
                norm_y = i / grid_h
                coords_list.append([norm_x, norm_y])
        
        coords = torch.tensor(coords_list, dtype=torch.float32).to(self.device)
        
        # Batch prediction
        batch_size = 1024
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(coords), batch_size):
                batch = coords[i:i + batch_size]
                batch_scores = self.model(batch).cpu().numpy()
                scores.extend(batch_scores)
        
        # Reshape to grid
        heat_map = np.array(scores).reshape(grid_h, grid_w)
        
        # Cache
        self._heat_map_cache = heat_map
        self._frame_size = (frame_width, frame_height)
        
        return heat_map
    
    def visualize_heat_map(
        self,
        frame: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay heat map on frame.
        
        Args:
            frame: Input frame (H, W, 3)
            alpha: Overlay transparency
        
        Returns:
            Visualization with heat map overlay
        """
        h, w = frame.shape[:2]
        
        # Get heat map
        heat_map = self.get_heat_map(w, h)
        
        # Resize to frame size
        heat_map_resized = cv2.resize(heat_map, (w, h))
        
        # Convert to color map (green = relevant, red = irrelevant)
        heat_map_colored = np.zeros((h, w, 3), dtype=np.uint8)
        heat_map_colored[:, :, 1] = (heat_map_resized * 255).astype(np.uint8)  # Green
        heat_map_colored[:, :, 2] = ((1 - heat_map_resized) * 255).astype(np.uint8)  # Red
        
        # Overlay
        result = cv2.addWeighted(frame, 1 - alpha, heat_map_colored, alpha, 0)
        
        # Add legend
        legend_h = 50
        legend = np.zeros((legend_h, w, 3), dtype=np.uint8)
        
        # Gradient bar
        gradient = np.linspace(0, 1, w).reshape(1, -1)
        gradient = np.repeat(gradient, 30, axis=0)
        
        legend_colored = np.zeros((30, w, 3), dtype=np.uint8)
        legend_colored[:, :, 1] = (gradient * 255).astype(np.uint8)
        legend_colored[:, :, 2] = ((1 - gradient) * 255).astype(np.uint8)
        
        legend[:30] = legend_colored
        
        # Text
        cv2.putText(legend, "Not Relevant", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(legend, "Relevant", (w - 100, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine
        final = np.vstack([result, legend])
        
        return final
