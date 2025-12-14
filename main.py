# main.py
import os
from multiprocessing import Process, Event, Value, Queue
import time
from utils import setup_worker_logger, reset_output_video
import multiprocessing as mp

import config
from workers.input_worker import input_worker
from workers.detect_worker import detect_worker
from workers.recog_worker import recog_worker
from workers.output_worker import output_worker
from workers.transform_worker import transform_worker

def main():
    logger = setup_worker_logger("MainProcess","logs/main.log")
    logger.info("Starting Logo Recognition Pipeline")
    logger.info(f"Using device: {config.DEVICE}")

    if not os.path.exists(config.VIDEO_PATH):
        logger.error(f"Video file not found: {config.VIDEO_PATH}")
        return
    if not os.path.exists(config.EMBED_DB_PATH):
        logger.error(f"Embedding database not found: {config.EMBED_DB_PATH}")
        logger.error("Please run 'python create_database.py' first to generate the database.")
        return

    # Reset output - delete existing files
    logger.info(f"Resetting output video: {config.OUTPUT_PATH}")
    reset_output_video(config.OUTPUT_PATH)

    # --- Create Queues and Sync Primitives ---
    input_to_detect_queue = Queue(maxsize=config.QUEUE_SIZE)
    detect_to_process_queue = Queue(maxsize=config.QUEUE_SIZE)
    process_to_recog_queue = Queue(maxsize=config.QUEUE_SIZE)
    recog_to_output_queue = Queue(maxsize=config.QUEUE_SIZE)

    start_event = Event()
    ready_counter = Value('i', 0)
    total_frames_output = Value('i', 0)

    # --- Create Worker Processes ---
    # Store processes in lists to manage shutdown by stage
    input_procs = []
    detect_procs = []
    transform_procs = []
    recog_procs = []
    output_procs = []

    # 1 Input worker
    p_input = Process(target=input_worker, name="InputWorker", args=(
        config.VIDEO_PATH,
        input_to_detect_queue,
        config.TARGET_FPS,
        start_event,
        ready_counter,
        None  # worker_id = None for single worker
    ))
    input_procs.append(p_input)

    # Detect workers
    num_detect_workers = config.NUM_YOLO_WORKERS
    for i in range(num_detect_workers):
        p_detect = Process(target=detect_worker, name=f"DetectWorker-{i+1}", args=(
            input_to_detect_queue,
            detect_to_process_queue,
            start_event,
            ready_counter,
            i + 1  # worker_id for unique logging
        ))
        detect_procs.append(p_detect)

    # Transform workers
    num_transform_workers = config.NUM_TRANSFORM_WORKERS
    for i in range(num_transform_workers):
        p_transform = Process(target=transform_worker, name=f"TransformWorker-{i+1}", args=(
            detect_to_process_queue,
            process_to_recog_queue,
            start_event,
            ready_counter,
            i + 1  # worker_id for unique logging
        ))
        transform_procs.append(p_transform)
    
    # Recog workers
    num_recog_workers = config.NUM_ARCFACE_WORKERS
    for i in range(num_recog_workers):
        p_recog = Process(target=recog_worker, name=f"RecogWorker-{i+1}", args=(
            process_to_recog_queue,
            recog_to_output_queue,
            start_event,
            ready_counter,
            i + 1  # worker_id for unique logging
        ))
        recog_procs.append(p_recog)

    # 1 Output worker
    p_output = Process(target=output_worker, name="OutputWorker", args=(
        recog_to_output_queue,
        config.OUTPUT_PATH,
        start_event,
        config.VIDEO_PATH,
        ready_counter,
        total_frames_output,
        None  # worker_id = None for single worker
    ))
    output_procs.append(p_output)
    
    all_processes = input_procs + detect_procs + transform_procs + recog_procs + output_procs
    num_workers = len(all_processes)

    # --- Start All Processes ---
    for p in all_processes:
        p.start()
        logger.info(f"Start process {p.name} (PID: {p.pid})")

    # --- Wait for all workers to be ready ---
    while True:
        with ready_counter.get_lock():
            if ready_counter.value >= num_workers:
                break
        time.sleep(0.1)

    # --- Start the Pipeline ---
    logger.info("All workers initialized and ready. Releasing start signal...")
    start_event.set()
    start_time = time.time()

    # NEW: Coordinated Shutdown Logic
    # 1. Wait for the input process to finish reading the video file.
    for p in input_procs:
        p.join()
    logger.info("Input worker has finished.")

    # 2. Signal all detection workers to stop and wait for them to finish.
    logger.info("Signaling detection workers to terminate...")
    for _ in range(num_detect_workers):
        input_to_detect_queue.put(None)
    for p in detect_procs:
        p.join()
    logger.info("Detection workers have finished.")

    # 3. Signal the transform workers to stop and wait for them.
    logger.info("Signaling transform workers to terminate...")
    for _ in range(num_transform_workers):
        detect_to_process_queue.put(None)
    for p in transform_procs:
        p.join()
    logger.info("Transform workers have finished.")

    # 4. Signal all recognition workers to stop and wait for them to finish.
    logger.info("Signaling recognition workers to terminate...")
    for _ in range(num_recog_workers):
        process_to_recog_queue.put(None)
    for p in recog_procs:
        p.join()
    logger.info("Recognition workers have finished.")

    # 5. Finally, signal the output worker to stop and wait for it.
    logger.info("Signaling output worker to terminate...")
    recog_to_output_queue.put(None)
    for p in output_procs:
        p.join()
    logger.info("Output worker has finished.")

    end_time = time.time()
    total_duration = end_time - start_time

    logger.info("All workers have finished.")
    logger.info(f"Total pipeline execution time: {total_duration:.2f} seconds.")

    if total_frames_output.value > 0 and total_duration > 0:
        logger.info(f"Total frames output: {total_frames_output.value}")
    logger.info(f"Output video saved to: {config.OUTPUT_PATH}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()