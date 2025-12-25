# VeroAllarme Data Directory

This directory contains runtime data for the VeroAllarme system.

## Structure

- `images/` - Alert images from security cameras
- `masks/` - User-defined polygon masks for each camera
- `heatmaps/` - Historical heat map data and visualizations

## Usage

These directories are automatically populated during system operation.
They are mounted as volumes in Docker for persistence.
