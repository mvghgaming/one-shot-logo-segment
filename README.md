# One-Shot Logo Recognition Pipeline

## Demo

[![Demo Video](https://img.youtube.com/vi/tcZ0v_jTTe0/0.jpg)](https://youtu.be/tcZ0v_jTTe0)

Watch the demo: [https://youtu.be/tcZ0v_jTTe0](https://youtu.be/tcZ0v_jTTe0)

## Overview

This project implements a one-shot logo recognition pipeline using YOLO for detection and EfficientNet-based embedding for recognition. The pipeline is modular, with separate workers for input, detection, pre-processing, recognition, and output.

## Features

- Batch video frame processing
- YOLO-based logo detection
- EfficientNet-based logo embedding and recognition
- Multiprocessing worker architecture
- Support for custom logo databases

## How to Run

**1. Create Required Folders:**
```bash
mkdir input video logs output
```
These folders are not included in the repository and must be created manually.

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Create Logo Database:**
```bash
python create_database.py
```
Place logo images in `support_data/` and run this to build the embedding database.

**4. Run Recognition Pipeline:**
```bash
# Single-process pipeline (with audio)
python pipeline.py

# Multi-process pipeline (faster)
python main.py
```
Configure video paths in `config.py` before running.

## Directory Structure

- `workers/`: All multiprocessing worker scripts (input, detect, transform, recognize, output)
- `support_data/`: Support images and masks for database creation
- `weights/`: Model weights and embedding database
- `utils.py`: Utility functions
- `model.py`: Model definition
- `config.py`: Configuration file
- `create_database.py`: Script to build the logo embedding database

## Files

**Main Scripts:**
- `main.py` - Multiprocessing pipeline with workers
- `pipeline.py` - Single-process pipeline (simpler, includes audio copying)
- `create_database.py` - Build logo embedding database from support images

**Core:**
- `config.py` - Configuration (paths, thresholds, device settings)
- `model.py` - LogoEncoder model (EfficientNet-based)
- `utils.py` - Helper functions (resize, drawing, embeddings)

## Configuration

- CPU: 12th gen intel(r) core(tm) i9-12900k
- Ram: 16GB
- GPU: NVIDIA GeForce RTX 3060 4VRAM