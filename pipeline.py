"""
pipeline.py - Single-file logo detection and recognition pipeline
Processes video frames sequentially: detect -> transform -> recognize -> output
"""
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
import subprocess
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO  # type: ignore[attr-defined]
from torch.amp.autocast_mode import autocast

import config
from model import LogoEncoder
from utils import load_embeddings, resize_with_padding, draw_on_frame, reset_output_video


def copy_audio(input_video, output_video_no_audio, output_video_with_audio):
    """Copy audio from input video to output video using ffmpeg"""
    try:
        print("\nCopying audio from original video...")

        # Use ffmpeg to copy audio from input and combine with processed video
        cmd = [
            'ffmpeg',
            '-i', output_video_no_audio,  # Processed video (no audio)
            '-i', input_video,            # Original video (with audio)
            '-c:v', 'copy',               # Copy video codec
            '-c:a', 'aac',                # Use AAC audio codec
            '-map', '0:v:0',              # Map video from first input
            '-map', '1:a:0?',             # Map audio from second input (optional)
            '-shortest',                   # Match shortest stream duration
            '-y',                          # Overwrite output file
            output_video_with_audio
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Audio copied successfully")
            # Remove temporary video without audio
            Path(output_video_no_audio).unlink()
            return True
        else:
            print(f"⚠ Warning: Could not copy audio (ffmpeg error)")
            print(f"  Video saved without audio: {output_video_no_audio}")
            return False

    except FileNotFoundError:
        print("⚠ Warning: ffmpeg not found. Install ffmpeg to preserve audio.")
        print(f"  Video saved without audio: {output_video_no_audio}")
        return False
    except Exception as e:
        print(f"⚠ Warning: Audio copy failed: {e}")
        print(f"  Video saved without audio: {output_video_no_audio}")
        return False


class LogoPipeline:
    def __init__(self):
        """Initialize all models and components for the pipeline"""
        print("=" * 60)
        print("Initializing Logo Detection & Recognition Pipeline")
        print("=" * 60)

        # Load YOLO detection model
        print(f"\n[1/3] Loading YOLO model from {config.YOLO_MODEL_PATH}")
        self.yolo_model = YOLO(config.YOLO_MODEL_PATH)

        # Move YOLO model to GPU (FP16 will be enabled via inference parameter)
        if torch.cuda.is_available():
            self.yolo_model.to('cuda')
            print("  ✓ YOLO model moved to GPU")

        # Store whether to use half precision
        self.use_half = torch.cuda.is_available()

        # Load recognition model
        print(f"\n[2/3] Loading recognition model from {config.RECOG_MODEL_PATH}")
        self.recog_model = LogoEncoder(config.EFFICIENTNET_WEIGHTS).to(config.DEVICE)
        checkpoint = torch.load(config.RECOG_MODEL_PATH, map_location=config.DEVICE)
        self.recog_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if torch.cuda.is_available():
            self.recog_model = self.recog_model.half()
            print("  ✓ Recognition model converted to FP16")

        self.recog_model.eval()

        # Load embedding database
        print(f"\n[3/3] Loading embedding database from {config.EMBED_DB_PATH}")
        db = load_embeddings(config.EMBED_DB_PATH)
        if not db:
            raise ValueError("Embedding database is empty or not found!")

        db_embeddings_np = np.array([e for e, _ in db])
        self.db_labels = [c for _, c in db]
        self.db_embeddings_tensor = torch.tensor(db_embeddings_np, dtype=torch.float32).to(config.DEVICE)

        if torch.cuda.is_available():
            self.db_embeddings_tensor = self.db_embeddings_tensor.half()

        self.db_embeddings_tensor = F.normalize(self.db_embeddings_tensor, dim=1)
        print(f"  ✓ Loaded {len(self.db_labels)} embeddings")

        # Define image transform for recognition
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60 + "\n")

    def detect(self, frame):
        """Run YOLO detection on a frame"""
        rgb_frame = frame[:, :, ::-1]  # BGR to RGB
        results = self.yolo_model(rgb_frame, verbose=False, half=self.use_half)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        masks_obj = results.masks

        return boxes, confs, masks_obj

    def transform(self, frame, boxes, confs, masks_obj):
        """Transform detections into tensors for recognition"""
        h_orig, w_orig = frame.shape[:2]

        bboxes_to_keep = []
        masks_for_drawing = []
        image_tensors = []
        mask_tensors = []

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

        # Process each detection
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            if conf < config.YOLO_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            bboxes_to_keep.append((x1, y1, x2, y2))

            # Prepare mask crop
            mask_crop = None
            if resized_masks is not None and i < len(resized_masks):
                mask_crop = resized_masks[i][y1:y2, x1:x2]
            masks_for_drawing.append(mask_crop)

            # Prepare image tensor for recognition
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            padded_img = resize_with_padding(pil_img, target_size=config.RECOG_IMAGE_SIZE)
            img_tensor = self.image_transform(padded_img)
            image_tensors.append(img_tensor)

            # Prepare mask tensor for recognition
            padded_mask_tensor = torch.ones(1, config.RECOG_IMAGE_SIZE, config.RECOG_IMAGE_SIZE)
            if mask_crop is not None:
                mask_pil = Image.fromarray(mask_crop, mode='L')
                padded_mask = resize_with_padding(mask_pil, target_size=config.RECOG_IMAGE_SIZE, fill_color=0)
                padded_mask_tensor = transforms.ToTensor()(padded_mask)
            mask_tensors.append(padded_mask_tensor)

        return image_tensors, mask_tensors, bboxes_to_keep, masks_for_drawing

    def recognize(self, image_tensors, mask_tensors):
        """Run logo recognition on transformed tensors"""
        if not image_tensors:
            return []

        all_labels = []
        batch_size = config.RECOG_BATCH_SIZE

        # Process in batches
        for i in range(0, len(image_tensors), batch_size):
            batch_image_tensors = image_tensors[i:i + batch_size]
            batch_mask_tensors = mask_tensors[i:i + batch_size]

            # Stack tensors
            batch_tensor = torch.stack(batch_image_tensors).to(config.DEVICE)
            batch_mask_tensor = torch.stack(batch_mask_tensors).to(config.DEVICE)

            # Run recognition
            with autocast('cuda', enabled=torch.cuda.is_available()):
                with torch.no_grad():
                    batch_embeddings = self.recog_model(batch_tensor, mask=batch_mask_tensor)
                    similarities = torch.mm(batch_embeddings, self.db_embeddings_tensor.T)
                    max_sims, max_indices = torch.max(similarities, dim=1)
                    max_sims_cpu = max_sims.cpu().numpy()
                    max_indices_cpu = max_indices.cpu().numpy()

                    for sim, idx in zip(max_sims_cpu, max_indices_cpu):
                        if sim > config.LOGO_SIMILARITY_THRESHOLD:
                            all_labels.append((self.db_labels[idx], float(sim)))
                        else:
                            all_labels.append(("Unknown", float(sim)))

        return all_labels

    def process_video(self, input_path, output_path):
        """Process entire video through the pipeline"""
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}\n")

        # Reset output - delete existing files
        reset_output_video(output_path)

        # Create temporary output path (without audio)
        output_path_temp = str(Path(output_path).with_suffix('')) + '_temp.mp4'

        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

        # Create output video writer (temporary file without audio)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
        out_writer = cv2.VideoWriter(output_path_temp, fourcc, fps, (width, height))

        # Processing statistics
        frame_id = 0
        start_time = time.time()
        total_logos_detected = 0
        total_logos_recognized = 0

        print("\nProcessing frames...")
        print("-" * 60)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start_time = time.time()

                # Step 1: Detect logos
                boxes, confs, masks_obj = self.detect(frame)

                # Step 2: Transform detections
                image_tensors, mask_tensors, bboxes, masks = self.transform(frame, boxes, confs, masks_obj)

                # Step 3: Recognize logos
                labels = self.recognize(image_tensors, mask_tensors)

                # Step 4: Draw results and write frame
                processed_frame = draw_on_frame(frame, bboxes, labels, masks)
                out_writer.write(processed_frame)

                # Update statistics
                frame_duration = time.time() - frame_start_time
                total_logos_detected += len(labels)
                recognized_logos = [label for label, _ in labels if label != "Unknown"]
                total_logos_recognized += len(recognized_logos)

                # Log progress
                if len(labels) > 0:
                    print(f"Frame {frame_id}: detected {len(labels)} logos, "
                          f"recognized {len(recognized_logos)} ({recognized_logos}), "
                          f"time {frame_duration:.4f}s")
                else:
                    print(f"Frame {frame_id}: no logos detected")

                frame_id += 1

                # Progress indicator every 30 frames
                if frame_id % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_id / elapsed
                    eta = (total_frames - frame_id) / fps_actual if fps_actual > 0 else 0
                    print(f"  Progress: {frame_id}/{total_frames} frames ({frame_id/total_frames*100:.1f}%), "
                          f"Speed: {fps_actual:.2f} FPS, ETA: {eta:.1f}s")

        finally:
            cap.release()
            out_writer.release()

        # Copy audio from original video
        audio_copied = copy_audio(input_path, output_path_temp, output_path)

        # Final statistics
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)
        print(f"Total frames processed: {frame_id}")
        print(f"Total logos detected: {total_logos_detected}")
        print(f"Total logos recognized: {total_logos_recognized}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average FPS: {frame_id / total_time:.2f}")
        if audio_copied:
            print(f"Output saved to: {output_path} (with audio)")
        else:
            print(f"Output saved to: {output_path_temp} (no audio)")
        print("=" * 60)


def main():
    """Main entry point"""
    # Initialize pipeline
    pipeline = LogoPipeline()

    # Process video
    pipeline.process_video(config.VIDEO_PATH, config.OUTPUT_PATH)


if __name__ == "__main__":
    main()
