"""Face alignment using affine transformation"""
import numpy as np
import cv2
import logging
from typing import Optional, Tuple


def align_face(
    src_img: np.ndarray,
    src_lm: np.ndarray,
    dst_lm: np.ndarray,
    output_size: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Align source face to destination face coordinates using affine transformation
    
    Args:
        src_img: Source image (numpy array, RGB or BGR)
        src_lm: Source landmarks (5, 2) numpy array
        dst_lm: Destination landmarks (5, 2) numpy array
        output_size: Output image size (width, height)
    
    Returns:
        Aligned face image or None if alignment fails
    """
    try:
        # Validate input shapes
        if src_lm.shape != (5, 2) or dst_lm.shape != (5, 2):
            logging.error(f"[FacePreserve][Align] Invalid landmark shapes: src={src_lm.shape}, dst={dst_lm.shape}")
            return None
        
        # Estimate affine transformation matrix
        # Using estimateAffinePartial2D for similarity transformation (rotation, scale, translation)
        M, inliers = cv2.estimateAffinePartial2D(
            src_lm.reshape(-1, 1, 2),
            dst_lm.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
        
        if M is None:
            logging.error("[FacePreserve][Align] Failed to estimate affine matrix")
            return None
        
        logging.info("[FacePreserve][Align] Affine transform estimated")
        
        # Apply affine transformation
        aligned = cv2.warpAffine(
            src_img,
            M,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned
        
    except Exception as e:
        logging.error(f"[FacePreserve][Align] Error during face alignment: {e}")
        return None


