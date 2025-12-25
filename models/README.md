# YOLO Models Directory

Place YOLO model weights here.

## Required Files

- `yolov8n.pt` - YOLOv8 Nano model (recommended for speed)
- `yolov8s.pt` - YOLOv8 Small model (optional, better accuracy)
- `yolov8m.pt` - YOLOv8 Medium model (optional, high accuracy)

## Download

Download YOLOv8 models from:
https://github.com/ultralytics/ultralytics

```bash
# Download automatically on first run, or manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Model Selection

- **yolov8n.pt** (6.3 MB) - Fast, good for CPU, ~80% accuracy
- **yolov8s.pt** (11.2 MB) - Balanced, better accuracy
- **yolov8m.pt** (25.9 MB) - High accuracy, requires GPU

VeroAllarme is optimized for conditional YOLO triggering, so even the nano model performs well.
