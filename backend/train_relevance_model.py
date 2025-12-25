"""
VeroAllarme - Relevance Model Training Script

This script trains the relevance mask model using self-supervised contrastive learning.

Training Phases:
1. Phase 1: Self-supervised contrastive learning (unlabeled data)
2. Phase 2: Fine-tuning segmentation head (pseudo-labeled with motion masks)
3. Phase 3 (optional): Active learning with user feedback

Usage:
    # Phase 1: Contrastive pre-training
    python train_relevance_model.py --phase contrastive --epochs 100
    
    # Phase 2: Segmentation fine-tuning
    python train_relevance_model.py --phase segmentation --pretrained checkpoint.pth --epochs 50
    
    # Full pipeline
    python train_relevance_model.py --phase both --epochs 150
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.relevance_mask import RelevanceMaskModel
from core.motion_detection import MotionDetector
from datasets.motion_dataset import (
    ContrastiveMotionDataset,
    SegmentationDataset,
    create_contrastive_dataloader
)


# ==================== Configuration ====================

class TrainingConfig:
    """Training configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration
        self.data_dir = "data/training/camera-events/Factory"
        self.output_dir = "backend/model_checkpoints"
        self.log_dir = "backend/logs/training"
        
        # Model
        self.feature_dim = 128
        self.num_classes = 2
        self.pretrained_backbone = True
        
        # Training
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.temperature = 0.07  # For contrastive loss
        
        # Optimization
        self.optimizer = "adam"
        self.scheduler = "cosine"
        self.warmup_epochs = 10
        
        # Data
        self.num_workers = 4
        self.max_sequences = None  # None = use all
        self.train_split = 0.8
        
        # Checkpointing
        self.save_every = 10
        self.validate_every = 5
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load from file if provided
        if config_path:
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_yaml(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}


# ==================== Loss Functions ====================

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    Used for contrastive learning (SimCLR)
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for scaling
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss
        
        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D)
            negative: Negative embeddings (B, D)
        
        Returns:
            Loss scalar
        """
        batch_size = anchor.size(0)
        
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        
        # Compute similarity scores
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # (B,)
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature  # (B,)
        
        # Compute loss (log softmax over positive and negative)
        logits = torch.stack([pos_sim, neg_sim], dim=1)  # (B, 2)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


# ==================== Training Functions ====================

def train_contrastive_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> float:
    """
    Train one epoch of contrastive learning
    
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (anchor, positive, negative) in enumerate(pbar):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        # Forward pass
        anchor_proj = model(anchor)
        positive_proj = model(positive)
        negative_proj = model(negative)
        
        # Compute loss
        loss = criterion(anchor_proj, positive_proj, negative_proj)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_segmentation_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Tuple[float, float]:
    """
    Train one epoch of segmentation
    
    Returns:
        (average loss, average IoU)
    """
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train Seg]")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device).long().squeeze(1)  # (B, H, W)
        
        # Forward pass
        logits = model(images)  # (B, 2, H, W)
        
        # Compute loss
        loss = criterion(logits, masks)
        
        # Compute IoU
        preds = torch.argmax(logits, dim=1)  # (B, H, W)
        intersection = ((preds == 1) & (masks == 1)).sum().item()
        union = ((preds == 1) | (masks == 1)).sum().item()
        iou = intersection / (union + 1e-6)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_iou += iou
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_iou


@torch.no_grad()
def validate_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """
    Validate segmentation model
    
    Returns:
        (average loss, average IoU)
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    for images, masks in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        masks = masks.to(device).long().squeeze(1)
        
        # Forward pass
        logits = model(images)
        
        # Compute loss
        loss = criterion(logits, masks)
        
        # Compute IoU
        preds = torch.argmax(logits, dim=1)
        intersection = ((preds == 1) & (masks == 1)).sum().item()
        union = ((preds == 1) | (masks == 1)).sum().item()
        iou = intersection / (union + 1e-6)
        
        total_loss += loss.item()
        total_iou += iou
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_iou


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    config: TrainingConfig,
    checkpoint_path: str,
    is_best: bool = False
):
    """Save training checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config.to_dict()
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # Save best model separately
    if is_best:
        best_path = Path(checkpoint_path).parent / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model: {best_path}")


# ==================== Main Training Loop ====================

def train_contrastive_phase(config: TrainingConfig, logger: SummaryWriter):
    """Phase 1: Self-supervised contrastive learning"""
    print("\n" + "=" * 80)
    print("PHASE 1: Self-Supervised Contrastive Learning")
    print("=" * 80)
    
    # Create dataset
    print(f"\nLoading contrastive dataset from {config.data_dir}...")
    dataloader = create_contrastive_dataloader(
        config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_sequences=config.max_sequences
    )
    
    print(f"✓ Dataset size: {len(dataloader.dataset)} sequences")
    print(f"✓ Batches per epoch: {len(dataloader)}")
    
    # Create model
    print("\nInitializing model...")
    model = RelevanceMaskModel(
        feature_dim=config.feature_dim,
        pretrained=config.pretrained_backbone,
        mode="contrastive"
    ).to(config.device)
    
    print(f"✓ Model created (device: {config.device})")
    
    # Loss and optimizer
    criterion = NTXentLoss(temperature=config.temperature)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )
    
    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss = train_contrastive_epoch(
            model, dataloader, criterion, optimizer, config.device, epoch
        )
        
        # Log
        logger.add_scalar("contrastive/train_loss", train_loss, epoch)
        logger.add_scalar("contrastive/lr", optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = Path(config.output_dir) / f"contrastive_epoch_{epoch}.pth"
            is_best = train_loss < best_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                config, str(checkpoint_path), is_best
            )
            
            if is_best:
                best_loss = train_loss
        
        # Step scheduler
        scheduler.step()
    
    # Save final model
    final_path = Path(config.output_dir) / "contrastive_final.pth"
    save_checkpoint(
        model, optimizer, config.num_epochs, train_loss,
        config, str(final_path)
    )
    
    print("\n✓ Phase 1 complete!")
    return model


def train_segmentation_phase(
    config: TrainingConfig,
    logger: SummaryWriter,
    pretrained_model: Optional[nn.Module] = None
):
    """Phase 2: Segmentation fine-tuning"""
    print("\n" + "=" * 80)
    print("PHASE 2: Segmentation Fine-Tuning")
    print("=" * 80)
    
    # Create dataset
    print(f"\nLoading segmentation dataset from {config.data_dir}...")
    motion_detector = MotionDetector()
    
    full_dataset = SegmentationDataset(
        config.data_dir,
        motion_detector=motion_detector,
        max_sequences=config.max_sequences
    )
    
    # Train/val split
    train_size = int(config.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"✓ Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create model
    print("\nInitializing model...")
    
    if pretrained_model is not None:
        # Use pre-trained encoder
        model = RelevanceMaskModel(
            feature_dim=config.feature_dim,
            mode="segmentation",
            pretrained=False
        )
        model.encoder.load_state_dict(pretrained_model.encoder.state_dict())
        print("✓ Loaded pre-trained encoder from Phase 1")
    else:
        model = RelevanceMaskModel(
            feature_dim=config.feature_dim,
            mode="segmentation",
            pretrained=config.pretrained_backbone
        )
    
    model = model.to(config.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )
    
    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    best_iou = 0.0
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss, train_iou = train_segmentation_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch
        )
        
        # Validate
        if epoch % config.validate_every == 0:
            val_loss, val_iou = validate_segmentation(
                model, val_loader, criterion, config.device
            )
            
            logger.add_scalar("segmentation/val_loss", val_loss, epoch)
            logger.add_scalar("segmentation/val_iou", val_iou, epoch)
            
            print(f"\n  Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Log
        logger.add_scalar("segmentation/train_loss", train_loss, epoch)
        logger.add_scalar("segmentation/train_iou", train_iou, epoch)
        logger.add_scalar("segmentation/lr", optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        
        # Save checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = Path(config.output_dir) / f"segmentation_epoch_{epoch}.pth"
            is_best = train_iou > best_iou
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                config, str(checkpoint_path), is_best
            )
            
            if is_best:
                best_iou = train_iou
        
        scheduler.step()
    
    # Save final model
    final_path = Path(config.output_dir) / "segmentation_final.pth"
    save_checkpoint(
        model, optimizer, config.num_epochs, train_loss,
        config, str(final_path)
    )
    
    print("\n✓ Phase 2 complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train VeroAllarme Relevance Model")
    
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--phase", type=str, default="both",
                       choices=["contrastive", "segmentation", "both"],
                       help="Training phase")
    parser.add_argument("--data-dir", type=str, help="Data directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained model")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = TrainingConfig(args.config)
    else:
        config = TrainingConfig()
    
    # Override with command-line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.device:
        config.device = args.device
    
    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.log_dir) / f"run_{timestamp}"
    logger = SummaryWriter(log_dir)
    
    # Save configuration
    config_save_path = Path(config.output_dir) / f"config_{timestamp}.yaml"
    config.save_to_yaml(str(config_save_path))
    print(f"✓ Saved configuration to {config_save_path}")
    
    # Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Execute training phases
    pretrained_model = None
    
    if args.phase in ["contrastive", "both"]:
        pretrained_model = train_contrastive_phase(config, logger)
    
    if args.phase in ["segmentation", "both"]:
        # Load pretrained model if specified
        if args.pretrained:
            print(f"\nLoading pretrained model from {args.pretrained}...")
            checkpoint = torch.load(args.pretrained, map_location=config.device)
            pretrained_model = RelevanceMaskModel(mode="contrastive")
            pretrained_model.load_state_dict(checkpoint["model_state_dict"])
        
        train_segmentation_phase(config, logger, pretrained_model)
    
    logger.close()
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nCheckpoints saved to: {config.output_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    main()
