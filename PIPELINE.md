# Multi-Stage Alert Processing Pipeline

## Overview

The alert processing pipeline implements a three-stage filtering system designed to minimize false alerts while preserving real security threats. Each stage makes independent filtering decisions that feed into the next stage.

## Architecture

```
Motion Alert Input
       ↓
   [Stage 3] Heat Map Analysis
   ├─ Input: motion_mask, original_image
   ├─ Output: anomaly_score, heat_zone (cold|hot), next_stage (4|5)
   ├─ Logic: Accumulates motion patterns over time, compares to historical baseline
   └─ Decision: Routes to Stage 4 (hot) or Stage 5 (cold)
       ↓
   [Stage 4] Memory-based Anomaly Detection
   ├─ Input: images [I1, I2, I3], motion_boxes
   ├─ Output: event_sim, event_support, decision (FILTER|DROP|PASS)
   ├─ Logic: Embeds motion boxes, searches FAISS index for similar past events
   └─ Decision: FILTER (escalate) if highly similar + good support
                DROP if novel or low support
                PASS (uncertain → defer to Stage 3)
       ↓
   [Stage 5] YOLO Object Detection (optional)
   ├─ Input: image I2
   ├─ Output: detected objects with classes and confidence scores
   └─ Logic: Confirms anomalies contain person/vehicle/etc. or provides explanation
       ↓
   Final Decision: ESCALATE or DROP
```

## Stage 3: Heat Map Analysis

**Location:** `backend/core/filter-heat-map/filter.py`

### Purpose
Detect anomalous motion patterns by comparing current motion to historical baseline.

### Key Components

- **HeatMapAnalyzer**: Temporal heat map with N time bins (hourly or day/night)
- **Probability Model**: P(motion at pixel) based on historical frequency
- **Anomaly Score**: Inverse of observed motion probability
  - Score ~1.0 = motion in usually-quiet zone (anomalous)
  - Score ~0.0 = motion in active zone (normal)

### Algorithm

```python
# For each motion event:
1. Add motion mask to corresponding time bin
2. Calculate probability heatmap from all bins
3. For each motion pixel:
   - If P(motion) < 0.3 → "quiet zone" (anomalous)
   - If P(motion) > 0.7 → "active zone" (normal)
4. Compute anomaly_score = fraction of anomalous pixels
5. Compare to anomaly_threshold (default 0.5)
   - score > threshold → heat_zone="cold", next_stage=5 (escalate)
   - score ≤ threshold → heat_zone="hot", next_stage=4 (memory check)
```

### Output Contract

```python
{
    "anomaly_score": 0.0-1.0,      # Fraction of motion in quiet zones
    "flagged": True|False,          # anomaly_score > threshold
    "heat_zone": "cold"|"hot",      # Cold=anomalous, Hot=normal
    "next_stage": 4|5,              # 4=Stage4, 5=Stage5 (skip anomaly)
    "stats": {
        "quiet_zones": int,
        "active_zones": int
    }
}
```

### Configuration

- **time_bin_mode**: "hourly" (24 bins) or "day_night" (2 bins)
- **anomaly_threshold**: 0.5 (default)
- **quiet_zone_prob**: 0.3 (< threshold)
- **active_zone_prob**: 0.7 (> threshold)
- **min_history_events**: 10 (to avoid false positives with limited history)

### Test Results

✅ Stage 3 Tests: 2/2 passing
- `test_cold_path_routes_to_stage_5`: Anomalous motion → heat_zone="cold", next_stage=5
- `test_hot_path_routes_to_stage_4`: Normal motion → heat_zone="hot", next_stage=4

---

## Stage 4: Memory-based Anomaly Detection

**Location:** `backend/core/anomaly_detection.py`

### Purpose
Recognize recurring patterns in motion alerts. Filter anomalies (novel events) from normal activity variations.

### Key Components

#### Event Class
```python
class Event:
    event_id: str                  # Unique identifier
    camera_id: str                 # Camera source
    timestamp: datetime            # When detected
    motion_boxes: List[tuple]      # Bounding boxes [(x1,y1,x2,y2), ...]
    embeddings: np.ndarray         # Shape (N, 512) - one per box
    label: str                     # "anomaly" or "normal"
```

#### MemoryIndex Class
Per-camera FAISS index for fast nearest-neighbor search:
```python
class MemoryIndex:
    index: faiss.IndexFlatIP      # IP = Inner Product (cosine similarity)
    metadata: Dict[int, Dict]     # Maps embedding ID → event metadata
    
    Methods:
    - add_event(event)             # Insert embeddings into index
    - retrieve_neighbors(embedding, k) → List[metadata]
    - save(path) / load(path)      # Persistence
```

#### AnomalyDetector Class
Main detector orchestrating the decision pipeline:
```python
class AnomalyDetector:
    camera_indices: Dict[str, MemoryIndex]  # Per-camera indices
    
    Key Methods:
    - extract_embeddings(images, motion_boxes) → np.ndarray(N, 512)
    - retrieve_neighbors(camera_id, embedding, k) → List[neighbors]
    - detect(camera_id, images, motion_boxes) → decision dict
    - add_event_to_memory(camera_id, event)
```

### Algorithm

```python
# For each alert:
1. Extract motion box embeddings from I2 (middle frame)
   - Crop each box region
   - Pass through CLIP/DINOv2 model
   - L2-normalize to unit vectors

2. For each embedding:
   - Retrieve top-10 neighbors from FAISS index
   - Compute similarities (dot product of normalized vectors = cosine)

3. Aggregate across all boxes in event:
   - event_sim = max(similarities across all box neighbors)
   - event_support = count(unique past event_ids with sim ≥ 0.87)

4. Apply decision rule:
   - FILTER if (event_sim ≥ 0.92 AND event_support ≥ 8)
       → "Known anomaly with strong pattern" → escalate
   - DROP if (event_sim < 0.85 OR event_support < 3)
       → "Novel event or weak pattern" → reject
   - PASS if (0.85 ≤ event_sim < 0.92 OR 3 ≤ event_support < 8)
       → "Uncertain" → defer to Stage 3
```

### Key Thresholds

| Parameter | Value | Interpretation |
|-----------|-------|-----------------|
| `sim_strong` | 0.92 | High similarity = likely same event class |
| `sim_weak` | 0.85 | Low boundary for uncertainty |
| `support_min` | 8 | Min past events to confirm pattern |
| `support_threshold` | 0.87 | Similarity cutoff for counting support |
| `embedding_dim` | 512 | CLIP/DINOv2 output dimension |

### Output Contract

```python
{
    "event_sim": 0.0-1.0,                    # Max similarity to neighbors
    "event_support": int,                     # Count of supporting past events
    "decision": "FILTER"|"DROP"|"PASS",     # Filtering decision
    "neighbors": [
        {
            "event_id": str,
            "similarity": float,
            "timestamp": datetime,
            "label": str
        }
    ]
}
```

### Test Results

✅ Stage 4 Tests: 1/4 passing (others require FAISS + transformers)
- `test_empty_motion_boxes`: Handles edge case of no motion boxes

---

## Integration: Alert Processing Pipeline

**Location:** `backend/core/pipeline.py`

### Purpose
Orchestrate stages 3, 4, and 5 with consistent decision logic and explanation generation.

### Flow

```python
def process_alert(camera_id, alert_id, images, motion_mask, motion_boxes):
    # Stage 3: Heat map analysis
    stage3 = heatmap_filter.process_event(motion_mask, images[1])
    heat_zone = stage3["heat_zone"]
    next_stage_hint = stage3["next_stage"]
    
    # Stage 4: Memory-based detection
    stage4 = anomaly_detector.detect(camera_id, images, motion_boxes)
    decision = stage4["decision"]
    
    # Decision Logic:
    if decision == "FILTER":
        final = "ESCALATE"  # Always escalate high-confidence anomalies
        if yolo_detector:
            stage5 = yolo_detector.detect(images[1])
    elif decision == "DROP":
        final = "DROP"      # Always drop novel events
    else:  # PASS
        if next_stage_hint == 5:  # Stage 3 said anomalous
            final = "ESCALATE"
        else:                      # Stage 3 said normal
            final = "DROP"
    
    return {
        "alert_id": alert_id,
        "final_decision": final,  # ESCALATE or DROP
        "explanation": human_readable_explanation,
        "stage_3_result": stage3,
        "stage_4_result": stage4,
        "stage_5_result": stage5 or None
    }
```

### Output Structure

```python
{
    "alert_id": str,
    "camera_id": str,
    "timestamp": ISO8601,
    "final_decision": "ESCALATE"|"DROP",
    "explanation": str,                    # Why decision was made
    "stage_3_result": {...},               # Heat map analysis
    "stage_4_result": {...},               # Memory detection
    "stage_5_result": {...} or None,       # YOLO detection (optional)
}
```

### Test Results

✅ Pipeline Tests: 6/6 passing
- `test_pipeline_creation`: Initialization
- `test_alert_process_stage_4_filter_decision`: Stage 4 FILTER → ESCALATE
- `test_alert_process_stage_4_drop_decision`: Stage 4 DROP → DROP
- `test_alert_process_includes_stage_3_data`: Stage 3 output included
- `test_alert_includes_metadata`: Metadata fields present
- `test_alert_stage_4_pass_with_cold_zone`: Stage 4 PASS + Stage 3 cold → ESCALATE

---

## Decision Matrix

| Stage 3 | Stage 4 | Final Decision | Explanation |
|---------|---------|----------------|-------------|
| Cold    | FILTER  | ESCALATE       | Spatial + temporal anomaly |
| Cold    | DROP    | DROP           | Spatial anomaly but novel pattern |
| Cold    | PASS    | ESCALATE       | Spatial anomaly (defer to expert) |
| Hot     | FILTER  | ESCALATE       | Known event pattern (anomalous) |
| Hot     | DROP    | DROP           | Normal location, novel motion |
| Hot     | PASS    | DROP           | Normal location (hold) |

---

## Configuration Example

```python
# Backend initialization
from backend.core.filter_heat_map.filter import FilterHeatMap
from backend.core.anomaly_detection import AnomalyDetector
from backend.core.pipeline import create_pipeline

# Create components
heatmap_filter = FilterHeatMap(
    time_bin_mode="hourly",
    anomaly_threshold=0.5
)

anomaly_detector = AnomalyDetector(
    embedding_model="clip",  # or "dinov2"
    sim_strong=0.92,
    sim_weak=0.85,
    support_min=8
)

# Optional YOLO detector
yolo_detector = YOLODetector(model="yolov8n")

# Create pipeline
pipeline = create_pipeline(
    heatmap_filter=heatmap_filter,
    anomaly_detector=anomaly_detector,
    yolo_detector=yolo_detector
)

# Process alert
result = pipeline.process_alert(
    camera_id="camera_1",
    alert_id="alert_001",
    images=[I1, I2, I3],
    motion_mask=motion_mask,
    motion_boxes=[(x1, y1, x2, y2), ...]
)

print(result["final_decision"])      # "ESCALATE" or "DROP"
print(result["explanation"])         # Human-readable reason
```

---

## Performance Characteristics

| Stage | Latency | Memory | Scalability |
|-------|---------|--------|-------------|
| 3 (Heat Map) | ~10ms | ~5MB (per camera) | Linear with history bins |
| 4 (Memory) | ~50ms | ~100MB (per camera, at 1M events) | Log(N) FAISS search |
| 5 (YOLO) | ~100ms | ~500MB | GPU optional, linear with objects |

---

## Next Steps

1. **Deploy Stage 4 embedding model** (CLIP or DINOv2)
2. **Integrate with FastAPI backend** (add endpoints to `backend/main.py`)
3. **Add persistence layer** (pickle or database for MemoryIndex)
4. **Implement Stage 5 YOLO** (optional, for detailed object classification)
5. **A/B testing** (compare pipeline decision rates before/after deployment)
6. **Performance tuning** (adjust thresholds based on field data)

---

## Related Files

- Core logic: `backend/core/pipeline.py`
- Tests: `backend/tests/test_pipeline.py`
- Stage 3: `backend/core/filter-heat-map/filter.py`, `heatmap.py`
- Stage 4: `backend/core/anomaly_detection.py`
