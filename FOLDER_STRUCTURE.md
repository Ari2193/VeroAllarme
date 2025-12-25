# Backend Structure: Organized by Stage

## Overview

The backend is now organized into logical stage-based folders, making the codebase cleaner and easier to maintain.

## Folder Structure

```
backend/
├── core/
│   ├── __init__.py
│   ├── motion_detection.py          # Stage 1-2: Motion detection (shared)
│   ├── pipeline.py                  # Orchestration (Stages 3-5)
│   │
│   ├── stage3/                      # Stage 3: Heat Map Analysis
│   │   ├── __init__.py
│   │   ├── filter.py                # FilterHeatMap class
│   │   ├── heatmap.py               # HeatMapAnalyzer class
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── conftest.py
│   │       └── test_filter_heat_map.py
│   │
│   ├── stage4/                      # Stage 4: Anomaly Detection (Memory)
│   │   ├── __init__.py
│   │   ├── anomaly_detection.py     # Event, MemoryIndex, AnomalyDetector classes
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── conftest.py
│   │       └── test_anomaly_detection.py
│   │
│   └── filter-heat-map/             # LEGACY (keep for backward compatibility)
│       ├── __init__.py
│       ├── filter.py
│       ├── heatmap.py
│       └── tests/
│           ├── __init__.py
│           ├── conftest.py
│           └── test_filter_heat_map.py
│
├── tests/
│   ├── test_pipeline.py             # Integration tests (Stages 3-5)
│   ├── test_anomaly_detection.py    # Legacy tests (deprecated, moved to stage4)
│   ├── test_motion_detection.py
│   └── ...
│
├── main.py                          # FastAPI app
├── config.py                        # Configuration
└── requirements-dev.txt
```

## What Moved Where

### Stage 3: Heat Map Analysis
**New location:** `backend/core/stage3/`

**Files:**
- `filter.py` - `FilterHeatMap` class (consumes motion masks, routes to next stage)
- `heatmap.py` - `HeatMapAnalyzer` class (historical pattern analysis)
- `tests/test_filter_heat_map.py` - Integration tests

**Imports:**
```python
from backend.core.stage3 import FilterHeatMap
from backend.core.stage3.heatmap import HeatMapAnalyzer
```

### Stage 4: Anomaly Detection
**New location:** `backend/core/stage4/`

**Files:**
- `anomaly_detection.py` - `Event`, `MemoryIndex`, `AnomalyDetector` classes
- `tests/test_anomaly_detection.py` - Unit and integration tests

**Imports:**
```python
from backend.core.stage4 import Event, MemoryIndex, AnomalyDetector
from backend.core.stage4 import AnomalyDetector  # Common use case
```

### Pipeline Orchestration
**Location:** `backend/core/pipeline.py`

**Key class:**
- `AlertProcessingPipeline` - Orchestrates Stages 3, 4, and optional Stage 5

**Imports:**
```python
from backend.core.pipeline import AlertProcessingPipeline, create_pipeline
```

## Usage Examples

### Running Tests

```bash
# All tests across all stages
pytest backend/tests/ backend/core/stage3/tests/ backend/core/stage4/tests/ -v

# Stage 3 only
pytest backend/core/stage3/tests/ -v

# Stage 4 only
pytest backend/core/stage4/tests/ -v

# Pipeline integration tests
pytest backend/tests/test_pipeline.py -v
```

### Using in Code

```python
from backend.core.stage3 import FilterHeatMap
from backend.core.stage4 import AnomalyDetector
from backend.core.pipeline import create_pipeline

# Initialize components
heatmap_filter = FilterHeatMap(camera_id="camera_1")
anomaly_detector = AnomalyDetector()

# Create pipeline
pipeline = create_pipeline(
    heatmap_filter=heatmap_filter,
    anomaly_detector=anomaly_detector
)

# Process alert
result = pipeline.process_alert(
    camera_id="camera_1",
    alert_id="alert_001",
    images=[I1, I2, I3],
    motion_mask=mask,
    motion_boxes=boxes
)

print(result["final_decision"])  # "ESCALATE" or "DROP"
```

## Backward Compatibility

The old `filter-heat-map/` folder is kept for backward compatibility. New code should use `stage3/` instead.

**Old imports (deprecated):**
```python
from backend.core.filter_heat_map.filter import FilterHeatMap  # ❌ Don't use
```

**New imports (preferred):**
```python
from backend.core.stage3 import FilterHeatMap  # ✅ Use this
```

## Benefits of This Structure

1. **Clear Organization**: Each stage has its own folder with related code
2. **Easy Navigation**: Find stage-specific code quickly
3. **Isolated Tests**: Test files live with their modules
4. **Scalability**: Adding Stage 5, 6, etc. is straightforward
5. **Better Imports**: Use `from backend.core.stage3 import ...` (cleaner than hyphenated paths)

## Test Results

✅ **30 tests passing, 4 skipped**

- Stage 3: 2 tests passing
- Stage 4: 3 tests passing, 2 skipped (require FAISS/transformers)
- Pipeline: 6 tests passing
- Legacy: 19 tests passing (motion_detection, etc.)

## Next Steps

1. **Optional Stage 5 (YOLO)**: Create `backend/core/stage5/` when needed
2. **Clean Up**: Eventually remove deprecated `filter-heat-map/` folder
3. **FastAPI Integration**: Connect pipeline to `backend/main.py` endpoints

---

**Last Updated:** 2025-12-25  
**Status:** ✅ All tests passing with new structure
