# config.py
import torch
import os

# --- Essential Paths ---
# BASE_DIR: Root directory of the project (where this config.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# VIDEO_PATH: Path to the input video file for processing
VIDEO_PATH = os.path.join(BASE_DIR, 'input', 'NBA-720p-10.91s.mp4')

# OUTPUT_PATH: Path to the output video file
OUTPUT_PATH = os.path.join(BASE_DIR, 'output', 'output_video.mp4')

# --- Model & Database Paths ---
# YOLO_MODEL_PATH: Path to the YOLO detection model weights
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'YOLO11m-seg_logo.pt')

# RECOG_MODEL_PATH: Path to the recognition (ArcFace) model weights
RECOG_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'arcface_logo_model_best.pth')

# EFFICIENTNET_WEIGHTS: Path to EfficientNet backbone weights
EFFICIENTNET_WEIGHTS = os.path.join(BASE_DIR, 'weights', 'efficientnet-b4-6ed6700e.pth')

# EMBED_DB_PATH: Path to the saved embedding database (pickle file)
EMBED_DB_PATH = os.path.join(BASE_DIR, 'weights', 'embedding_db.pkl')

# --- Support Data for Database Creation ---
# SUPPORT_DIR: Directory containing support images for each class (used for database creation)
SUPPORT_DIR = os.path.join(BASE_DIR, 'support_data', 'support_yolo')

# MASK_DIR: Directory containing mask images for each class (optional, used for database creation)
MASK_DIR = os.path.join(BASE_DIR, 'support_data', 'mask_yolo')

# --- Processing Parameters ---
# DEVICE: Device to use for computation (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TARGET_FPS: Target frames per second for input video processing
TARGET_FPS = 30

# RECOG_IMAGE_SIZE: Image size to which logos are resized for recognition
RECOG_IMAGE_SIZE = 380

# --- Thresholds ---
# YOLO_CONF_THRESHOLD: Confidence threshold for YOLO detections
YOLO_CONF_THRESHOLD = 0.5

# LOGO_SIMILARITY_THRESHOLD: Confidence threshold for logo recognition (cosine similarity)
LOGO_SIMILARITY_THRESHOLD = 0.5

# --- Worker & Queue Configs ---
# NUM_YOLO_WORKERS: Number of parallel YOLO detection worker processes
NUM_YOLO_WORKERS = 1

# NUM_PREPROCESS_WORKERS: Number of parallel preprocess worker processes (for cropping/masking)
NUM_PREPROCESS_WORKERS = 1

# NUM_ARCFACE_WORKERS: Number of parallel recognition worker processes
NUM_ARCFACE_WORKERS = 1

# YOLO_BATCH_SIZE: Batch size for YOLO inference
YOLO_BATCH_SIZE = 4

# RECOG_BATCH_SIZE: Batch size for recognition inference
RECOG_BATCH_SIZE = 16

# QUEUE_SIZE: Maximum number of items in each multiprocessing queue
QUEUE_SIZE = 100 # Max items in each queue

# --- Censoring Settings ---
# CENSOR_ENABLED: Whether to censor recognized logos (True) or show annotations (False)
CENSOR_ENABLED = False

# CENSOR_SHAPE: Shape to use for censoring ("mask" for exact logo shape, "bbox" for rectangle)
CENSOR_SHAPE = "mask"  # Options: "mask" or "bbox"

# CENSOR_COLOR: Color for censoring in BGR format (B, G, R)
CENSOR_COLOR = (0, 255, 0)  # Green

# MAX_MASK_SIZE: Maximum mask area (in pixels) to censor. Logos with larger masks won't be censored.
# Set to None to disable size filtering (censor all recognized logos regardless of size)
MAX_MASK_SIZE = 10000  # 10KB = 10,000 pixels

# OUTLINE_ENABLED: Draw outline around recognized logo masks
OUTLINE_ENABLED = True

# OUTLINE_COLOR: Color for outline in BGR format (B, G, R)
OUTLINE_COLOR = (0, 255, 0)  # Green

# OUTLINE_THICKNESS: Thickness of the outline in pixels
OUTLINE_THICKNESS = 3