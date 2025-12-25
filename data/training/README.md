# Training Data Directory

This directory contains historical camera event data for training and testing the VeroAllarme system.

## Structure

```
training/
└── camera-events/
    ├── Factory/         # Factory camera events
    ├── Field/           # Field camera events
    ├── Gate and rode/   # Gate and road camera events
    └── Trees/           # Trees area camera events
```

## Dataset Format

Each event directory contains 2-3 sequential images:
- `0.jpg` - First frame
- `1.jpg` - Second frame (motion detected)
- `2.jpg` - Third frame (optional)

## Usage

This data can be used for:
1. **Training anomaly detection models** - Learn normal vs. abnormal patterns
2. **Testing motion detection algorithms** - Validate frame differencing
3. **Heat map generation** - Build historical motion patterns
4. **YOLO trigger optimization** - Determine when to invoke object detection
5. **Ground truth labeling** - Manual classification for supervised learning

## Adding More Data

To add new training data:
```bash
mkdir -p data/training/camera-events/[camera_name]/[timestamp]
# Copy event images (0.jpg, 1.jpg, 2.jpg) to the timestamp directory
```

## Statistics

Run this command to get dataset statistics:
```bash
find data/training/camera-events -name "*.jpg" | wc -l  # Total images
find data/training/camera-events -type d -mindepth 2 | wc -l  # Total events
```

## Notes

- This directory is in `.gitignore` by default (large files)
- Consider using Git LFS if you need to version control this data
- For production, store training data in cloud storage (S3/MinIO)
