"""
Stage 3: Heat Map Analysis (Historical Patterns)

This module provides heat map-based anomaly detection by analyzing
spatial patterns of motion over time.

Main classes:
- FilterHeatMap: Consumes motion masks, updates history, calculates anomaly scores
- HeatMapAnalyzer: Core heat map accumulation and probability calculation

Usage:
    from backend.core.filter_heat_map import FilterHeatMap
    
    filter = FilterHeatMap(camera_id="camera_1")
    result = filter.process_event(motion_mask, original_image)
    print(result["heat_zone"])  # "cold" or "hot"
    print(result["next_stage"])  # 4 or 5
"""

try:
    from .filter import FilterHeatMap
    from .heatmap import HeatMapAnalyzer
except ImportError:
    # Handle case where relative import fails (e.g., during test discovery)
    try:
        from filter import FilterHeatMap
        from heatmap import HeatMapAnalyzer
    except ImportError:
        pass

__all__ = ["FilterHeatMap", "HeatMapAnalyzer"]
