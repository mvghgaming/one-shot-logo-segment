# detect_worker.py
import time
import queue
import numpy as np
from utils import setup_worker_logger
from ultralytics import YOLO  # type: ignore[attr-defined]
import config
import torch

def detect_worker(input_queue, output_queue, start_event, ready_counter, worker_id):
    # Initialize logger for this worker process with unique ID
    logger = setup_worker_logger("DetectWorker", "logs/detect_worker.log", worker_id)
    logger.info(f"Detection worker {worker_id} starting model...")

    # Load YOLO model using the path from config
    yolo_model = YOLO(config.YOLO_MODEL_PATH)

    # Move YOLO model to GPU (don't manually convert to half, YOLO handles this internally)
    if torch.cuda.is_available():
        yolo_model.to('cuda')
        logger.info("YOLO model moved to GPU")
    else:
        logger.info("CUDA not available, using CPU")

    logger.info("YOLO model loaded successfully.")

    # Mark this worker as ready by incrementing the shared counter
    with ready_counter.get_lock():
        ready_counter.value += 1
        
    # Wait for the main process to signal start
    logger.info("Waiting for start signal...")
    start_event.wait()
    logger.info("Start signal received. Begin detection loop...")

    # Set batch size for YOLO inference
    batch_size = config.YOLO_BATCH_SIZE
    stop_signal_received = False
    
    # Initialize counters for performance logging
    total_frames_processed = 0
    total_processing_time = 0.0

    # Main detection loop
    while not stop_signal_received:
        frames_to_process = []        # List to hold frames for batch processing
        original_data_list = []       # List to keep original data for output

        # Try to fill a batch from the input queue
        for _ in range(batch_size):
            try:
                data = input_queue.get_nowait()
                if data is None:
                    # Stop signal received from main process
                    stop_signal_received = True
                    break
                
                frame_id, frame = data
                rgb_frame = frame[:, :, ::-1] # Convert BGR to RGB for YOLO

                frames_to_process.append(rgb_frame)
                original_data_list.append((frame_id, frame))

            except queue.Empty:
                # No more frames available right now
                break
        
        if frames_to_process:
            # Run YOLO inference on the batch with half precision if CUDA is available
            batch_start_time = time.time()
            results_batch = yolo_model(frames_to_process, verbose=False, half=torch.cuda.is_available())
            batch_end_time = time.time()
            
            num_processed = len(frames_to_process)
            total_frames_processed += num_processed
            duration = batch_end_time - batch_start_time
            total_processing_time += duration

            # Pass raw detection results to the next worker
            for i, results in enumerate(results_batch):
                frame_id, frame = original_data_list[i]

                # Extract detection boxes, confidences, and masks
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                masks_obj = results.masks # Raw ultralytics mask object

                # Put results into the output queue for the next worker
                output_queue.put((frame_id, frame, boxes, confs, masks_obj))

        elif not stop_signal_received:
            # If no frames to process, sleep briefly to avoid busy waiting
            time.sleep(0.001)

    # Log summary statistics when worker finishes
    logger.info(f"Detection worker {worker_id} finished.")
    logger.info(f"Total frames processed: {total_frames_processed}")
    if total_frames_processed > 0:
        logger.info(f"Average processing time per frame: {total_processing_time / total_frames_processed:.4f} seconds")
    logger.info(f"Total processing time: {total_processing_time:.4f} seconds")