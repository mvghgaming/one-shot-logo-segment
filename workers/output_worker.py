# output_worker.py
import time
from utils import setup_worker_logger, draw_on_frame
import cv2
import numpy as np
import subprocess
import os

def output_worker(input_queue, output_path, start_event, video_path, ready_counter, total_frames_output=None, worker_id=None):
    # Initialize logger for this worker process
    logger = setup_worker_logger("OutputWorker", "logs/output_worker.log", worker_id)
    logger.info("Output worker starting...")

    # Initialization: Get video properties and set up writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file for data: {video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Write to a temporary file first, then merge audio later
    temp_output_path = output_path.replace('.mp4', '_temp_no_audio.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    out_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    logger.info(f"Output video writer initialized for {temp_output_path} with size ({width}x{height}), fps={fps}")

    # Mark this worker as ready
    with ready_counter.get_lock():
        ready_counter.value += 1

    # Wait for start signal from main process
    logger.info("Waiting for start signal...")
    start_event.wait()
    logger.info("Start signal received. Begin output loop...")

    # Initialize buffers and counters
    buffer = {}
    next_frame_to_write = 0
    total_frames_written = 0
    total_frames_received = 0
    total_time = 0.0
    start_write_time = None

    # Main Loop: Receive data and write frames in order
    while True:
        # Wait for data from the input queue (blocking)
        data = input_queue.get()

        # Exit loop if 'None' signal is received
        if data is None:
            logger.info("Received 'None' signal. Flushing buffer.")
            break

        if start_write_time is None:
            start_write_time = time.time()  

        # Unpack and buffer the received data
        start_time = time.time()
        frame_id, frame, bboxes, labels, masks = data

        if frame is not None:
            total_frames_received += 1
            buffer[frame_id] = (frame, bboxes, labels, masks)
        
        # Write frames in order if available in buffer
        while next_frame_to_write in buffer:
            item = buffer.pop(next_frame_to_write)
            frame_data, bboxes_data, labels_data, masks_data = item
            processed_frame = draw_on_frame(frame_data, bboxes_data, labels_data, masks_data)
            out_writer.write(processed_frame)
            total_frames_written += 1

            # Log each frame output
            num_detections = len(bboxes_data) if bboxes_data is not None else 0
            logger.info(f"Frame {next_frame_to_write} written ({num_detections} detections)")

            next_frame_to_write += 1

            end_time = time.time()
            duration = end_time - start_time
            total_time += duration

    # Final Flush: Write any remaining out-of-order frames in the buffer
    logger.info(f"Flushing {len(buffer)} remaining frames from buffer...")
    sorted_remaining_keys = sorted(buffer.keys())
    for frame_idx in sorted_remaining_keys:
        item = buffer.pop(frame_idx)
        frame_data, bboxes_data, labels_data, masks_data = item
        processed_frame = draw_on_frame(frame_data, bboxes_data, labels_data, masks_data)
        out_writer.write(processed_frame)
        total_frames_written += 1

        # Log each flushed frame
        num_detections = len(bboxes_data) if bboxes_data is not None else 0
        logger.info(f"Frame {frame_idx} flushed and written ({num_detections} detections)")
        
    # Finalization and Logging
    end_write_time = time.time()
    out_writer.release()
    write_duration = end_write_time - start_write_time if start_write_time is not None else 0

    logger.info(f"Temporary video (no audio) saved to {temp_output_path}")
    logger.info(f"Total time writing video: {write_duration:.4f} seconds")
    logger.info(f"Total frames received: {total_frames_received}")
    logger.info(f"Total frames written to video file: {total_frames_written}")
    if total_frames_received > 0:
        logger.info(f"Total processing time: {total_time:.4f} seconds")
        if total_frames_written > 0:
            logger.info(f"Average processing time per frame: {total_time / total_frames_written:.4f} seconds")
    if write_duration > 0 and total_frames_written > 0:
        logger.info(f"Average output FPS: {total_frames_written / write_duration:.2f}")

    # Merge audio from original video using ffmpeg
    logger.info("Merging audio from original video using ffmpeg...")
    try:
        # Use ffmpeg to copy audio from original video and merge with new video
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', temp_output_path,  # Input: processed video (no audio)
            '-i', video_path,        # Input: original video (with audio)
            '-map', '0:v:0',         # Map video from first input
            '-map', '1:a:0?',        # Map audio from second input (optional with ?)
            '-c:v', 'copy',          # Copy video codec (no re-encoding)
            '-c:a', 'aac',           # Encode audio to AAC
            '-shortest',             # Match shortest stream duration
            '-y',                    # Overwrite output file if exists
            output_path
        ]

        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        logger.info(f"Successfully merged audio. Final video saved to {output_path}")

        # Clean up temporary file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            logger.info(f"Removed temporary file {temp_output_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr}")
        logger.warning(f"Video without audio is available at {temp_output_path}")
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install ffmpeg to merge audio.")
        logger.warning(f"Video without audio is available at {temp_output_path}")
    except Exception as e:
        logger.error(f"Error during audio merge: {str(e)}")
        logger.warning(f"Video without audio is available at {temp_output_path}")

    # Optionally update shared counter for total frames written
    if total_frames_output is not None:
        with total_frames_output.get_lock():
            total_frames_output.value = total_frames_written