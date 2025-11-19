"""Face blending with feather mask"""
import numpy as np
import cv2
import logging
from typing import Tuple


def feather_blend(
    base: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    feather: int = 10
) -> np.ndarray:
    """
    Blend patch into base image using feathered alpha mask
    
    Args:
        base: Base image (numpy array, RGB)
        patch: Patch image to blend (numpy array, RGB)
        mask: Binary mask (numpy array, 0-255, single channel)
        feather: Feather radius for Gaussian blur (default: 10)
    
    Returns:
        Blended image (numpy array, RGB)
    """
    try:
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Normalize mask to 0-1 range
        mask_normalized = mask.astype(np.float32) / 255.0
        
        # Apply Gaussian blur for feathering
        if feather > 0:
            # Use odd kernel size
            kernel_size = feather * 2 + 1
            mask_feathered = cv2.GaussianBlur(
                mask_normalized,
                (kernel_size, kernel_size),
                sigmaX=feather / 3.0,
                sigmaY=feather / 3.0
            )
        else:
            mask_feathered = mask_normalized
        
        # Expand mask to 3 channels if needed
        if len(base.shape) == 3 and base.shape[2] == 3:
            mask_3d = mask_feathered[:, :, np.newaxis]
        else:
            mask_3d = mask_feathered
        
        # Ensure images have same shape
        if base.shape != patch.shape:
            logging.warning(f"[FacePreserve][Blend] Shape mismatch: base={base.shape}, patch={patch.shape}")
            # Resize patch to match base
            patch = cv2.resize(patch, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Blend: base * (1-mask) + patch * mask
        result = (
            base.astype(np.float32) * (1 - mask_3d) +
            patch.astype(np.float32) * mask_3d
        ).astype(np.uint8)
        
        logging.info("[FacePreserve][Blend] Feather blending complete")
        
        return result
        
    except Exception as e:
        logging.error(f"[FacePreserve][Blend] Error during blending: {e}")
        # Return base image on error
        return base


