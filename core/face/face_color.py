"""Face color matching using LAB color space"""
import numpy as np
import cv2
import logging
from typing import Optional


def color_match(
    source_face: np.ndarray,
    target_face: np.ndarray
) -> Optional[np.ndarray]:
    """
    Match skin tone from source face to target face using LAB color space
    
    Algorithm:
    1. Convert both images to LAB color space
    2. Compute mean and std for each channel
    3. Standardize source face → re-scale to target statistics
    4. Clip to 0~255 range
    5. Convert back to RGB
    
    Args:
        source_face: Source face image (numpy array, RGB or BGR)
        target_face: Target face image (numpy array, RGB or BGR)
    
    Returns:
        Color-corrected source face (numpy array, RGB) or None if failure
    """
    try:
        # Ensure images are in BGR format for OpenCV
        if len(source_face.shape) != 3 or source_face.shape[2] != 3:
            logging.error(f"[FacePreserve][Color] Invalid source_face shape: {source_face.shape}")
            return None
        
        if len(target_face.shape) != 3 or target_face.shape[2] != 3:
            logging.error(f"[FacePreserve][Color] Invalid target_face shape: {target_face.shape}")
            return None
        
        # Convert to BGR if needed (assume RGB input)
        source_bgr = source_face.copy()
        target_bgr = target_face.copy()
        
        # If images are RGB, convert to BGR
        # Check by comparing first pixel (RGB usually has higher R, BGR has higher B)
        # For safety, assume RGB and convert
        if source_bgr.dtype != np.uint8:
            source_bgr = source_bgr.astype(np.uint8)
        if target_bgr.dtype != np.uint8:
            target_bgr = target_bgr.astype(np.uint8)
        
        # Convert RGB to BGR (OpenCV expects BGR)
        # If already BGR, this will be a no-op
        source_bgr = cv2.cvtColor(source_bgr, cv2.COLOR_RGB2BGR)
        target_bgr = cv2.cvtColor(target_bgr, cv2.COLOR_RGB2BGR)
        
        # Convert to LAB color space
        source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Compute mean and std for each channel
        source_mean = np.mean(source_lab, axis=(0, 1))
        source_std = np.std(source_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))
        
        # Avoid division by zero
        source_std = np.where(source_std < 1e-6, 1.0, source_std)
        target_std = np.where(target_std < 1e-6, 1.0, target_std)
        
        # Standardize source: (source - source_mean) / source_std
        source_normalized = (source_lab - source_mean) / source_std
        
        # Re-scale to target: source_normalized * target_std + target_mean
        source_matched = source_normalized * target_std + target_mean
        
        # Clip to valid LAB range
        # L: 0-100, A: -127 to 127, B: -127 to 127
        source_matched[:, :, 0] = np.clip(source_matched[:, :, 0], 0, 100)
        source_matched[:, :, 1] = np.clip(source_matched[:, :, 1], -127, 127)
        source_matched[:, :, 2] = np.clip(source_matched[:, :, 2], -127, 127)
        
        # Convert back to BGR
        source_matched_uint8 = source_matched.astype(np.uint8)
        result_bgr = cv2.cvtColor(source_matched_uint8, cv2.COLOR_LAB2BGR)
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        # Clip to 0-255 range (safety check)
        result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
        
        logging.info("[FacePreserve][Color] Skin tone matched")
        
        return result_rgb
        
    except Exception as e:
        logging.error(f"[FacePreserve][Color] Error during color matching: {e}")
        return None

