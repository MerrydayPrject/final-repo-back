"""Face preservation pipeline with color matching"""
import numpy as np
import cv2
import logging
from typing import Union
from PIL import Image

from core.face.face_landmark import extract_landmarks
from core.face.face_align import align_face
from core.face.face_color import color_match
from core.face.face_blend import feather_blend


def preserve_face_pipeline(
    original_face_patch: Union[Image.Image, np.ndarray],
    gemini_output_img: Union[Image.Image, np.ndarray]
) -> np.ndarray:
    """
    Preserve original face in Gemini output using landmark-based alignment, 
    color matching, and blending
    
    Args:
        original_face_patch: Original face patch (PIL Image or numpy array, RGB)
        gemini_output_img: Gemini generated image (PIL Image or numpy array, RGB)
    
    Returns:
        Final blended image as numpy array (RGB)
    """
    logging.info("[FacePreserve] Start")
    
    # Convert PIL Images to numpy arrays (RGB format)
    if isinstance(original_face_patch, Image.Image):
        if original_face_patch.mode == 'RGBA':
            face_patch_rgb = original_face_patch.convert('RGB')
        else:
            face_patch_rgb = original_face_patch.convert('RGB')
        src_img = np.array(face_patch_rgb)
    else:
        src_img = original_face_patch.copy()
        # Convert BGR to RGB if needed
        if len(src_img.shape) == 3 and src_img.shape[2] == 3:
            if src_img.dtype == np.uint8:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    
    if isinstance(gemini_output_img, Image.Image):
        dst_img = np.array(gemini_output_img.convert('RGB'))
    else:
        dst_img = gemini_output_img.copy()
        # Convert BGR to RGB if needed
        if len(dst_img.shape) == 3 and dst_img.shape[2] == 3:
            if dst_img.dtype == np.uint8:
                dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
    
    # Step 1: Extract landmarks from both images
    # InsightFace expects BGR format
    src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    dst_img_bgr = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
    
    src_landmarks = extract_landmarks(src_img_bgr)
    dst_landmarks = extract_landmarks(dst_img_bgr)
    
    # Check if both landmarks are available
    if src_landmarks is None or dst_landmarks is None:
        logging.info("[FacePreserve] Skipped (no landmarks)")
        return dst_img
    
    logging.info("[FacePreserve] Landmarks OK")
    
    try:
        # Step 2: Align face
        output_size = (dst_img.shape[1], dst_img.shape[0])  # (width, height)
        aligned_face = align_face(src_img, src_landmarks, dst_landmarks, output_size)
        
        if aligned_face is None:
            logging.warning("[FacePreserve] Face alignment failed - returning original Gemini output")
            return dst_img
        
        logging.info("[FacePreserve] Affine OK")
        
        # Step 3: Extract target region from Gemini output
        # Calculate bounding box around dst_landmarks
        x_coords = dst_landmarks[:, 0]
        y_coords = dst_landmarks[:, 1]
        
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))
        
        # Add padding (20% of face size)
        face_width = x_max - x_min
        face_height = y_max - y_min
        padding_x = int(face_width * 0.2)
        padding_y = int(face_height * 0.2)
        
        # Ensure ROI is within image bounds
        h, w = dst_img.shape[:2]
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        # Ensure valid region
        if x_max <= x_min or y_max <= y_min:
            logging.warning("[FacePreserve] Invalid target region - skipping color matching")
            aligned_face_color = aligned_face
        else:
            # Extract target region
            target_region = dst_img[y_min:y_max, x_min:x_max]
            
            if target_region.size == 0:
                logging.warning("[FacePreserve] Target region extraction failed - skipping color matching")
                aligned_face_color = aligned_face
            else:
                # Step 4: Color match
                # Extract corresponding region from aligned_face for color matching
                # Use the same bounding box coordinates (adjusted for full image size)
                aligned_x_min = max(0, int(x_min))
                aligned_y_min = max(0, int(y_min))
                aligned_x_max = min(aligned_face.shape[1], int(x_max))
                aligned_y_max = min(aligned_face.shape[0], int(y_max))
                
                if aligned_x_max > aligned_x_min and aligned_y_max > aligned_y_min:
                    aligned_face_region = aligned_face[aligned_y_min:aligned_y_max, aligned_x_min:aligned_x_max]
                    
                    # Resize to match target_region size if needed
                    if aligned_face_region.shape[:2] != target_region.shape[:2]:
                        aligned_face_region = cv2.resize(
                            aligned_face_region,
                            (target_region.shape[1], target_region.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    aligned_face_color_region = color_match(aligned_face_region, target_region)
                    
                    if aligned_face_color_region is None:
                        logging.warning("[FacePreserve] Color matching failed - using aligned face without color correction")
                        aligned_face_color = aligned_face
                    else:
                        logging.info("[FacePreserve] Color OK")
                        
                        # Create full-size color-corrected face
                        aligned_face_color = aligned_face.copy()
                        
                        # Resize color-corrected region back to original size if needed
                        if aligned_face_color_region.shape[:2] != (aligned_y_max - aligned_y_min, aligned_x_max - aligned_x_min):
                            aligned_face_color_region = cv2.resize(
                                aligned_face_color_region,
                                (aligned_x_max - aligned_x_min, aligned_y_max - aligned_y_min),
                                interpolation=cv2.INTER_LINEAR
                            )
                        
                        # Place color-corrected region back
                        aligned_face_color[aligned_y_min:aligned_y_max, aligned_x_min:aligned_x_max] = aligned_face_color_region
                else:
                    logging.warning("[FacePreserve] Invalid aligned face region - skipping color matching")
                    aligned_face_color = aligned_face
        
        # Step 5: Create blending mask (ellipse shape)
        mask = np.zeros((dst_img.shape[0], dst_img.shape[1]), dtype=np.float32)
        
        # Calculate ellipse parameters from landmarks
        face_center = np.mean(dst_landmarks, axis=0)
        
        # Calculate distances from center to landmarks
        distances = np.linalg.norm(dst_landmarks - face_center, axis=1)
        max_distance = np.max(distances)
        
        # Ellipse axes (1.5x for safety margin)
        axes = (int(max_distance * 1.5), int(max_distance * 1.3))
        center = (int(face_center[0]), int(face_center[1]))
        
        # Draw filled ellipse
        cv2.ellipse(
            mask,
            center,
            axes,
            0,  # angle
            0,  # startAngle
            360,  # endAngle
            1.0,  # color (white)
            -1  # thickness (filled)
        )
        
        # Step 6: Feather blend
        # Convert mask to 0-255 uint8 for feather_blend
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        final_img = feather_blend(dst_img, aligned_face_color, mask_uint8, feather=10)
        
        logging.info("[FacePreserve] Blend OK")
        logging.info("[FacePreserve] Complete")
        
        return final_img
        
    except Exception as e:
        logging.error(f"[FacePreserve] Error in face preservation pipeline: {e}")
        # Return original Gemini output on error
        return dst_img
