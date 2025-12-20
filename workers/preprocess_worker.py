# preprocess_worker.py
import time
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

from utils import setup_worker_logger, resize_with_padding
import config

def preprocess_worker(input_queue, output_queue, start_event, ready_counter, worker_id):
    # Initialize logger for this worker process with unique ID
    logger = setup_worker_logger("PreprocessWorker", "logs/preprocess_worker.log", worker_id)
    logger.info(f"Preprocess worker {worker_id} starting...")

    # Define image transform for recognition model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get original video frame size for mask resizing
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Mark this worker as ready
    with ready_counter.get_lock():
        ready_counter.value += 1

    # Wait for start signal from main process
    logger.info("Waiting for start signal...")
    start_event.wait()
    logger.info("Start signal received. Begin pre-processing loop...")

    # Initialize statistics counters
    total_frames_processed = 0
    total_processing_time = 0.0

    # Main pre-processing loop
    while True:
        data = input_queue.get()
        if data is None:
            logger.info("Stop signal received.")
            break

        # Start timing for this frame
        start_time = time.time()

        frame_id, frame, boxes, confs, masks_obj = data

        # Prepare output lists for this frame
        bboxes_to_keep, masks_for_drawing = [], []
        image_tensors, mask_tensors = [], []

        if boxes is not None and confs is not None:
            # Resize masks to original frame size if available
            resized_masks = None
            if masks_obj is not None:
                mask_tensor_cpu = masks_obj.data.cpu()
                resized_masks_list = [
                    cv2.resize(m.numpy(), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                    for m in mask_tensor_cpu
                ]
                if resized_masks_list:
                    resized_masks = np.stack(resized_masks_list, axis=0)

            # Process each detection in the frame
            for i, (box, conf) in enumerate(zip(boxes, confs)):
                if conf < config.YOLO_CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                bboxes_to_keep.append((x1, y1, x2, y2))

                # Prepare mask crop for this detection
                mask_crop = None
                if resized_masks is not None and i < len(resized_masks):
                    mask_crop = resized_masks[i][y1:y2, x1:x2]
                masks_for_drawing.append(mask_crop)

                # Prepare image tensor for recognition
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                padded_img = resize_with_padding(pil_img, target_size=config.RECOG_IMAGE_SIZE)
                img_tensor = transform(padded_img)
                image_tensors.append(img_tensor)

                # Prepare mask tensor for recognition
                padded_mask_tensor = torch.ones(1, config.RECOG_IMAGE_SIZE, config.RECOG_IMAGE_SIZE)
                if mask_crop is not None:
                    mask_pil = Image.fromarray(mask_crop, mode='L')
                    padded_mask = resize_with_padding(mask_pil, target_size=config.RECOG_IMAGE_SIZE, fill_color=0)
                    padded_mask_tensor = transforms.ToTensor()(padded_mask)
                mask_tensors.append(padded_mask_tensor)

        # Put processed data into output queue for recognition worker
        output_queue.put((frame_id, frame, image_tensors, mask_tensors, bboxes_to_keep, masks_for_drawing))

        # End timing and update statistics
        end_time = time.time()
        duration = end_time - start_time
        total_processing_time += duration
        total_frames_processed += 1

    # Log summary statistics when worker finishes
    logger.info(f"Preprocess worker {worker_id} finished.")
    logger.info(f"Total frames processed: {total_frames_processed}")
    if total_frames_processed > 0:
        logger.info(f"Average pre-processing time per frame: {total_processing_time / total_frames_processed:.4f} seconds")
    logger.info(f"Total pre-processing time: {total_processing_time:.4f} seconds")
