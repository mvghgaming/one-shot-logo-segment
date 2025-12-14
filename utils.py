# utils.py
import os
import pickle
import logging
from PIL import Image, ImageOps
import cv2
import numpy as np

def setup_worker_logger(name, log_file, worker_id=None):
    """
    Set up a logger for a worker process with both file and console handlers.
    Ensures no duplicate handlers are attached and creates worker-specific log files.
    
    Args:
        name: Base name for the logger
        log_file: Base log file path
        worker_id: Optional worker ID for unique naming (e.g., "1", "2", etc.)
    """
    # Create unique logger name and file path for each worker
    if worker_id is not None:
        unique_name = f"{name}-{worker_id}"
        # Insert worker ID into filename before extension
        base_name, ext = os.path.splitext(log_file)
        unique_log_file = f"{base_name}_{worker_id}{ext}"
    else:
        unique_name = name
        unique_log_file = log_file
    
    logger = logging.getLogger(unique_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter('[%(asctime)s] [%(name)-15s] [%(levelname)-8s] - %(message)s')

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Ensure log directory exists
    os.makedirs(os.path.dirname(unique_log_file), exist_ok=True)

    # File handler for logging to file
    fh = logging.FileHandler(unique_log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler for logging to stdout
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def resize_with_padding(pil_img, target_size=380, fill_color=(128, 128, 128)):
    """
    Resize a PIL image to a target size while maintaining aspect ratio by padding.
    Uses NEAREST for masks and BICUBIC for other images.
    """
    w, h = pil_img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Use NEAREST for masks to preserve binary values, BICUBIC otherwise
    resample_method = Image.Resampling.NEAREST if pil_img.mode == 'L' else Image.Resampling.BICUBIC
    resized_img = pil_img.resize((new_w, new_h), resample_method)
    
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    
    # Calculate padding for each side
    padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
    padded_img = ImageOps.expand(resized_img, padding, fill=fill_color)
    return padded_img

def apply_color_censor(frame, x1, y1, x2, y2, mask, color, shape):
    """
    Fill region with solid color using mask shape or bounding box.

    Args:
        frame: The image frame (modified in-place)
        x1, y1, x2, y2: Bounding box coordinates
        mask: Optional segmentation mask (binary array)
        color: Fill color in BGR format (tuple)
        shape: "mask" to use mask shape, "bbox" for rectangle
    """
    roi_h = y2 - y1
    roi_w = x2 - x1
    if roi_h <= 0 or roi_w <= 0:
        return

    if shape == "mask" and mask is not None and mask.size > 0:
        # Fill only within mask shape
        resized_mask = cv2.resize(mask.astype(np.uint8), (roi_w, roi_h))
        roi = frame[y1:y2, x1:x2]
        roi[resized_mask > 0] = color
    else:
        # Fill entire bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)  # -1 = filled

def load_embeddings(path):
    """
    Load embeddings from a pickle file if it exists.
    Returns None if the file does not exist.
    """
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def save_embeddings(embeddings, path):
    """
    Save embeddings to a pickle file, creating directories if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(embeddings, f)

def reset_output_video(output_path):
    """
    Delete existing output video file if it exists.
    This ensures a clean start for new processing.

    Args:
        output_path: Path to the output video file
    """
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print(f"[RESET] Deleted existing output: {output_path}")
        except Exception as e:
            print(f"[RESET] Warning: Could not delete {output_path}: {e}")

    # Also clean up temporary files
    temp_path = output_path.replace('.mp4', '_no_audio.mp4')
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            print(f"[RESET] Deleted temporary file: {temp_path}")
        except Exception as e:
            print(f"[RESET] Warning: Could not delete {temp_path}: {e}")

def draw_mask_outline(frame, x1, y1, x2, y2, mask, color, thickness):
    """
    Draw an outline around the mask contours.

    Args:
        frame: The image frame (modified in-place)
        x1, y1, x2, y2: Bounding box coordinates
        mask: Segmentation mask (binary array)
        color: Outline color in BGR format
        thickness: Outline thickness in pixels
    """
    if mask is None or mask.size == 0:
        # If no mask, draw rectangle outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return

    roi_h = y2 - y1
    roi_w = x2 - x1
    if roi_h <= 0 or roi_w <= 0:
        return

    # Resize mask to fit the ROI
    resized_mask = cv2.resize(mask.astype(np.uint8), (roi_w, roi_h))

    # Find contours in the mask
    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the frame (offset by bounding box position)
    for contour in contours:
        # Offset contour points to absolute frame coordinates
        contour_shifted = contour + np.array([x1, y1])
        cv2.drawContours(frame, [contour_shifted], -1, color, thickness)

def draw_on_frame(frame, bboxes, labels, masks):
    """
    Draw bounding boxes, labels, scores, and optional masks on a frame.
    For recognized logos: applies color censoring if CENSOR_ENABLED is True.
    For unknown logos: hidden (not drawn).

    Args:
        frame: The image frame (numpy array).
        bboxes: List of bounding boxes.
        labels: List of (label, score) tuples.
        masks: List of mask arrays or None.
    Returns:
        The frame with drawn results.
    """
    from config import CENSOR_ENABLED, CENSOR_SHAPE, CENSOR_COLOR, MAX_MASK_SIZE, OUTLINE_ENABLED, OUTLINE_COLOR, OUTLINE_THICKNESS

    # Make a copy to avoid modifying the original frame if it's used elsewhere
    output_frame = frame.copy()
    if bboxes is not None and labels is not None:
        # Ensure masks is a list of the same length, even if it's all Nones
        if masks is None:
            masks = [None] * len(bboxes)

        for bbox, label_info, mask in zip(bboxes, labels, masks):
            # Check if label_info is a tuple (label, score)
            if isinstance(label_info, (list, tuple)) and len(label_info) == 2:
                label, score = label_info
            else:
                # Handle cases where label format might be unexpected
                label, score = "Error", 0.0

            x1, y1, x2, y2 = map(int, bbox)

            # Check mask size filter (applies regardless of CENSOR_ENABLED)
            should_process = True
            if MAX_MASK_SIZE is not None and mask is not None and mask.size > 0:
                mask_area = np.sum(mask > 0)  # Count non-zero pixels
                if mask_area > MAX_MASK_SIZE:
                    should_process = False  # Skip logos with large masks

            # Only process if mask size is within limits
            if not should_process:
                continue  # Skip this logo entirely

            # When censoring is enabled: only process recognized logos, skip unknown
            if CENSOR_ENABLED:
                if label != "Unknown":
                    # Apply color censoring to recognized logos
                    apply_color_censor(output_frame, x1, y1, x2, y2, mask, CENSOR_COLOR, CENSOR_SHAPE)

                    # Draw outline around the censored area
                    if OUTLINE_ENABLED:
                        draw_mask_outline(output_frame, x1, y1, x2, y2, mask, OUTLINE_COLOR, OUTLINE_THICKNESS)
                # Skip drawing unknown logos (they won't appear)
            else:
                # When censoring is disabled: draw annotations for all logos
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

                # Draw bounding box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

                # Prepare text for label and score
                text = f"{label} ({score:.2f})"
                # Put text above the bounding box
                cv2.putText(output_frame, text, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw mask if available and valid
                if mask is not None and mask.size > 0:
                    roi = output_frame[y1:y2, x1:x2]
                    if roi.shape[0] > 0 and roi.shape[1] > 0:
                        try:
                            # Resize mask to fit the ROI
                            resized_mask = cv2.resize(mask.astype(np.uint8), (roi.shape[1], roi.shape[0]))
                            colored_mask = np.zeros_like(roi)
                            # Ensure resized_mask is boolean for indexing
                            colored_mask[resized_mask > 0] = color
                            # Blend the mask with the ROI
                            blended_roi = cv2.addWeighted(roi, 1.0, colored_mask, 0.4, 0)
                            output_frame[y1:y2, x1:x2] = blended_roi
                        except cv2.error as e:
                            # Ignore errors from invalid mask/ROI
                            pass
    return output_frame