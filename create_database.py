# create_database.py
import os
import torch
from typing import cast
from PIL import Image
from torchvision import transforms
import logging

import config
from model import LogoEncoder
from utils import resize_with_padding, save_embeddings

# Set up basic logging configuration for this script
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def register_logo(image_path, class_name, model, mask_path=None):
    """
    Encodes a single logo image (optionally with a mask) and returns its embedding and class name.
    """
    model.eval()
    try:
        # Load and pad the logo image
        image = Image.open(image_path).convert('RGB')
        padded_img = resize_with_padding(image, target_size=config.RECOG_IMAGE_SIZE)
        
        mask_tensor = None
        # If a mask exists, load and pad it; otherwise, warn and use the full image
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            padded_mask = resize_with_padding(mask, target_size=config.RECOG_IMAGE_SIZE, fill_color=0)
            mask_tensor = transforms.ToTensor()(padded_mask).unsqueeze(0).to(config.DEVICE)
        else:
            logging.warning(f"No mask for {class_name}, using full image.")

        # Transform the image for model input
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = cast(torch.Tensor, transform(padded_img))
        image_tensor = img_tensor.unsqueeze(0).to(config.DEVICE)
        
        # Generate embedding using the model (with or without mask)
        with torch.no_grad():
            embedding = model(image_tensor, mask=mask_tensor).cpu().numpy().squeeze()
        
        logging.info(f"Successfully registered logo: {class_name} from {os.path.basename(image_path)}")
        return (embedding, class_name)
        
    except Exception as e:
        # Log any errors encountered during registration
        logging.error(f"Error registering logo {class_name} from {image_path}: {e}")
        return None

def register_support_folder():
    """
    Scans the support and mask folders, encodes all logos, and saves the embedding database.
    """
    logging.info("Starting database creation...")
    
    # Check if the support directory exists
    if not os.path.exists(config.SUPPORT_DIR):
        logging.error(f"Support directory not found: {config.SUPPORT_DIR}")
        return

    # Initialize the recognition model and load weights
    recog_model = LogoEncoder(config.EFFICIENTNET_WEIGHTS).to(config.DEVICE)
    checkpoint = torch.load(config.RECOG_MODEL_PATH, map_location=config.DEVICE)
    recog_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    recog_model.eval()
    
    # Remove existing database if present
    if os.path.exists(config.EMBED_DB_PATH):
        logging.warning(f"Existing database found at {config.EMBED_DB_PATH}. It will be overwritten.")
        os.remove(config.EMBED_DB_PATH)

    db = []
    # List all class directories in the support folder
    class_dirs = [d for d in os.listdir(config.SUPPORT_DIR) if os.path.isdir(os.path.join(config.SUPPORT_DIR, d))]

    if not class_dirs:
        logging.error("No class folders found in the support directory.")
        return

    # Iterate through each class directory
    for class_name in class_dirs:
        class_path = os.path.join(config.SUPPORT_DIR, class_name)
        mask_class_path = os.path.join(config.MASK_DIR, class_name)
        
        # List all image files in the class directory
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            mask_path = None
            
            # If a mask directory exists for this class, look for a matching mask file
            if os.path.exists(mask_class_path):
                base_name = os.path.splitext(img_file)[0]
                # Look for a mask with a matching name and supported extension
                for ext in ['.png', '.jpg']:
                    potential_mask = os.path.join(mask_class_path, f"{base_name}{ext}")
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break
            
            # Register the logo and add its embedding to the database
            result = register_logo(img_path, class_name, recog_model, mask_path)
            if result:
                db.append(result)

    # Save the database if any embeddings were registered
    if db:
        save_embeddings(db, config.EMBED_DB_PATH)
        logging.info(f"Successfully created database with {len(db)} entries from {len(class_dirs)} classes.")
        logging.info(f"Database saved to: {config.EMBED_DB_PATH}")
    else:
        logging.error("Database creation failed. No logos were registered.")

if __name__ == "__main__":
    # Run the database creation process if this script is executed
    register_support_folder()