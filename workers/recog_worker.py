# recog_worker.py
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.amp.autocast_mode import autocast

import config
from model import LogoEncoder
from utils import load_embeddings, setup_worker_logger

# Identify a batch of logos by comparing embeddings to the database
def identify_logo_batch(image_tensors, mask_tensors, model, db_embeddings, db_labels):
    if not image_tensors:
        return []
    model.eval()
    # Stack image and mask tensors for batch processing
    batch_tensor = torch.stack(image_tensors).to(config.DEVICE)
    batch_mask_tensor = torch.stack(mask_tensors).to(config.DEVICE)
    
    # Use FP16 inference
    with autocast('cuda', enabled=torch.cuda.is_available()):
        with torch.no_grad():
            # Get embeddings for the batch
            batch_embeddings = model(batch_tensor, mask=batch_mask_tensor)
            # Compute cosine similarity with database embeddings
            similarities = torch.mm(batch_embeddings, db_embeddings.T)
            max_sims, max_indices = torch.max(similarities, dim=1)
            max_sims_cpu = max_sims.cpu().numpy()
            max_indices_cpu = max_indices.cpu().numpy()
            results = []
            for sim, idx in zip(max_sims_cpu, max_indices_cpu):
                # If similarity above threshold, assign label; else "Unknown"
                if sim > config.RECOG_CONF_THRESHOLD:
                    results.append((db_labels[idx], float(sim)))
                else:
                    results.append(("Unknown", float(sim)))
    return results

def recog_worker(input_queue, output_queue, start_event, ready_counter, worker_id):
    # Initialize logger for this worker process with unique ID
    logger = setup_worker_logger("RecogWorker", "logs/recog_worker.log", worker_id)
    logger.info(f"Recognition worker {worker_id} starting model and database...")

    # Load recognition model and weights
    recog_model = LogoEncoder(config.EFFICIENTNET_WEIGHTS).to(config.DEVICE)
    checkpoint = torch.load(config.RECOG_MODEL_PATH, map_location=config.DEVICE)
    recog_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Enable FP16 for 2x speed improvement
    if torch.cuda.is_available():
        recog_model = recog_model.half()  # Convert to FP16
        logger.info("Recognition model converted to FP16 (half precision)")
    
    recog_model.eval()
    logger.info("Recognition model loaded.")

    # Load embedding database
    db = load_embeddings(config.EMBED_DB_PATH)
    if not db:
        logger.error("Embedding database is empty or not found!")
        return
    db_embeddings_np = np.array([e for e, _ in db])
    db_labels = [c for _, c in db]
    db_embeddings_tensor = torch.tensor(db_embeddings_np, dtype=torch.float32).to(config.DEVICE)
    
    # Convert database embeddings to FP16 too
    if torch.cuda.is_available():
        db_embeddings_tensor = db_embeddings_tensor.half()
        
    db_embeddings_tensor = F.normalize(db_embeddings_tensor, dim=1)
    logger.info(f"Database loaded with {len(db_labels)} embeddings.")

    # Mark this worker as ready
    with ready_counter.get_lock():
        ready_counter.value += 1

    # Wait for start signal from main process
    logger.info("Waiting for start signal...")
    start_event.wait()
    logger.info("Start signal received. Begin recognition loop...")
    
    # Initialize statistics counters
    total_frames_processed = 0
    total_processing_time = 0.0
    total_logos_processed = 0

    # Main recognition loop
    while True:
        data = input_queue.get()
        if data is None:
            logger.info("Stop signal received.")
            break
            
        # Start timing for this frame
        start_time = time.time()

        frame_id, frame, image_tensors, mask_tensors, bboxes, masks_for_drawing = data

        if not image_tensors:
            # No detections for this frame; output empty results
            output_queue.put((frame_id, frame, [], [], []))
            total_frames_processed += 1
            logger.info(f"Frame {frame_id}: no logos detected")
            continue

        all_labels = []
        batch_size = config.RECOG_BATCH_SIZE
        # Process detections in batches for efficiency
        for i in range(0, len(image_tensors), batch_size):
            batch_image_tensors = image_tensors[i:i + batch_size]
            batch_mask_tensors = mask_tensors[i:i + batch_size]
            batch_labels = identify_logo_batch(
                batch_image_tensors, batch_mask_tensors, 
                recog_model, db_embeddings_tensor, db_labels
            )
            all_labels.extend(batch_labels)

        total_logos_processed += len(all_labels)

        # Put recognition results into output queue for next worker
        output_queue.put((frame_id, frame, bboxes, all_labels, masks_for_drawing))

        # End timing and update statistics
        end_time = time.time()
        duration = end_time - start_time
        total_processing_time += duration
        total_frames_processed += 1

        # Log each frame's processing details
        recognized_logos = [label for label, _ in all_labels if label != "Unknown"]
        unknown_count = len(all_labels) - len(recognized_logos)
        logger.info(f"Frame {frame_id}: detected {len(all_labels)} logos, recognized {len(recognized_logos)} ({recognized_logos}), unknown {unknown_count}, time {duration:.4f}s")

    # Log summary statistics when worker finishes
    logger.info(f"Recognition worker {worker_id} finished.")
    logger.info(f"Total frames processed: {total_frames_processed}")
    logger.info(f"Total logos processed: {total_logos_processed}")
    if total_frames_processed > 0 and total_processing_time > 0:
        logger.info(f"Average recognition time per frame: {total_processing_time / total_frames_processed:.4f} seconds")
        logger.info(f"Average logos per second: {total_logos_processed / total_processing_time:.1f}")
    logger.info(f"Total recognition processing time: {total_processing_time:.4f} seconds")