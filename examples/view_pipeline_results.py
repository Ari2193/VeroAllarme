"""
Quick viewer to display sample visualizations from the pipeline.

This script shows a few example annotated images from the pipeline execution.
"""

import cv2
from pathlib import Path


def display_event_visualizations(event_prefix: str, output_dir: Path):
    """Display all visualizations for a single event."""
    print(f"\n{'='*60}")
    print(f"Event: {event_prefix}")
    print(f"{'='*60}")
    
    stages = ['stage3', 'stage4', 'final']
    
    for stage in stages:
        img_path = output_dir / f"{event_prefix}_{stage}.jpg"
        if img_path.exists():
            print(f"  ✓ {stage}: {img_path}")
        else:
            print(f"  ✗ {stage}: Not found")


def main():
    output_dir = Path("data/pipeline_outputs")
    
    if not output_dir.exists():
        print("❌ No pipeline outputs found. Run examples/run_pipeline_on_data.py first.")
        return
    
    print("\n" + "="*80)
    print("Pipeline Visualization Summary")
    print("="*80)
    
    # Get all unique event prefixes
    stage3_files = sorted(output_dir.glob("*_stage3.jpg"))
    
    print(f"\nTotal events processed: {len(stage3_files)}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Show details for first 3 events
    print("\n" + "-"*80)
    print("Sample Event Visualizations")
    print("-"*80)
    
    for stage3_file in stage3_files[:3]:
        event_prefix = stage3_file.stem.replace('_stage3', '')
        display_event_visualizations(event_prefix, output_dir)
    
    print("\n" + "="*80)
    print("📁 All visualizations saved in:", output_dir.absolute())
    print("📄 Full report available at:", output_dir / "pipeline_report.txt")
    print("📊 JSON results available at:", output_dir / "pipeline_results.json")
    print("="*80 + "\n")
    
    # List all events
    print("\nAll processed events:")
    for i, stage3_file in enumerate(stage3_files, 1):
        event_prefix = stage3_file.stem.replace('_stage3', '')
        print(f"  {i:2d}. {event_prefix}")


if __name__ == "__main__":
    main()
