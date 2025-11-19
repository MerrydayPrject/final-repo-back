"""Background removal utility for transparent base image generation"""
import numpy as np
import logging
from typing import Optional
from PIL import Image

from config.hf_segformer import FACE_MASK_IDS, CLOTH_MASK_IDS, BODY_MASK_IDS


def remove_background(person_img: Image.Image, parsing_mask: np.ndarray) -> Image.Image:
    """
    Remove background from person image using parsing mask, creating transparent RGBA image
    
    Args:
        person_img: Person image (PIL Image, RGB)
        parsing_mask: Parsing mask from SegFormer (numpy array with label IDs)
    
    Returns:
        Image.Image: Transparent base image (RGBA format, HxWx4)
                     - Foreground (face + body + cloth): alpha=255
                     - Background: alpha=0
    """
    try:
        # Create foreground mask: face_mask + body_mask + cloth_mask
        face_mask = np.isin(parsing_mask, FACE_MASK_IDS).astype(np.uint8)
        body_mask = np.isin(parsing_mask, BODY_MASK_IDS).astype(np.uint8)
        cloth_mask = np.isin(parsing_mask, CLOTH_MASK_IDS).astype(np.uint8)
        
        # Combine all foreground masks (OR operation)
        foreground_mask = np.clip(face_mask + body_mask + cloth_mask, 0, 1).astype(np.uint8)
        
        # Convert to alpha channel (0-255 range)
        alpha_channel = foreground_mask * 255
        
        # Convert person image to numpy array (RGB)
        person_array = np.array(person_img.convert("RGB"))
        
        # Create RGBA image
        h, w = person_array.shape[:2]
        transparent_array = np.zeros((h, w, 4), dtype=np.uint8)
        transparent_array[:, :, :3] = person_array  # RGB channels
        transparent_array[:, :, 3] = alpha_channel  # Alpha channel
        
        # Create PIL Image from array
        transparent_img = Image.fromarray(transparent_array, mode='RGBA')
        
        logging.info("[BG] Transparent base image created")
        print("[BG] Transparent base image created")
        
        return transparent_img
        
    except Exception as e:
        logging.error(f"[BG] Error creating transparent base image: {e}")
        print(f"[BG] Error creating transparent base image: {e}")
        # Fallback: return original image converted to RGBA (fully opaque)
        return person_img.convert("RGBA")

