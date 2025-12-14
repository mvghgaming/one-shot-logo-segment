# input_worker.py
import cv2
import time
from utils import setup_worker_logger

def input_worker(video_path, frame_queue, target_fps, start_event, ready_counter, worker_id=None):
    # Initialize logger for this worker process
    logger = setup_worker_logger("InputWorker", "logs/input_worker.log", worker_id)
    logger.info("Input worker start...")

    # Mark this worker as ready by incrementing the shared counter
    with ready_counter.get_lock():
        ready_counter.value += 1

    # Wait for the main process to signal start
    logger.info("Waiting for start signal...")
    start_event.wait()
    logger.info("Start signal received. Begin video reader...")

    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        # Signal failure to the next worker by putting None in the queue
        frame_queue.put(None)
        return

    # Initialize statistics and frame counters
    frame_id = 0
    total_frames = 0
    total_time = 0.0   

    # Calculate the interval between frames to match the target FPS
    frame_interval = 1.0 / target_fps

    # Main loop: read frames from the video and put them in the queue
    while True:
        loop_start_time = time.time()
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            # End of video stream reached
            logger.info("End of video stream reached.")
            break

        # Put frame data into the queue for the next worker
        frame_queue.put((frame_id, frame))

        # Increment counters
        frame_id += 1
        total_frames += 1

        # Measure time taken for this loop iteration
        elapsed = time.time() - loop_start_time
        total_time += elapsed

        # Sleep if needed to maintain the target FPS
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Release the video capture resource
    cap.release()
    # Log summary statistics when worker finishes
    logger.info("Input worker finished.")
    logger.info(f"Total frames processed: {total_frames}")
    if total_frames > 0:
        logger.info(f"Total time reading and queueing frames: {total_time:.4f} seconds")
        logger.info(f"Average time per frame: {total_time / total_frames:.4f} seconds")