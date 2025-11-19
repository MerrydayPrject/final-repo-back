"""Face landmark detection using InsightFace"""
import numpy as np
import cv2
from typing import Optional
import logging

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None
    logging.warning("InsightFace not available. Install with: pip install insightface")

# Global face analyzer instance (lazy initialization)
_face_analyzer = None


def _get_face_analyzer():
    """Get or initialize InsightFace FaceAnalysis instance"""
    global _face_analyzer
    if _face_analyzer is None and INSIGHTFACE_AVAILABLE:
        try:
            _face_analyzer = FaceAnalysis(name="antelopev2")
            _face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            logging.info("[FacePreserve][Landmark] InsightFace model loaded (antelopev2)")
        except Exception as e:
            logging.error(f"[FacePreserve][Landmark] Failed to initialize InsightFace: {e}")
            return None
    return _face_analyzer


def extract_landmarks(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract 5-keypoint landmarks from image using InsightFace
    
    Args:
        img: Input image as numpy array (BGR format)
    
    Returns:
        (5, 2) numpy array of keypoints in order:
        [left_eye, right_eye, nose, mouth_left, mouth_right]
        Returns None if no face detected
    """
    if not INSIGHTFACE_AVAILABLE:
        logging.warning("[FacePreserve][Landmark] InsightFace not available")
        return None
    
    face_analyzer = _get_face_analyzer()
    if face_analyzer is None:
        return None
    
    try:
        # InsightFace expects BGR format
        # Input should already be in BGR (converted in face_pipeline.py)
        img_bgr = img
        
        # Detect faces
        faces = face_analyzer.get(img_bgr)
        
        if len(faces) == 0:
            logging.debug("[FacePreserve][Landmark] No faces detected")
            return None
        
        # Use the first detected face
        face = faces[0]
        
        # Extract 5 keypoints from InsightFace kps
        # InsightFace kps format: (5, 2) array
        # Order: left_eye, right_eye, nose, mouth_left, mouth_right
        if hasattr(face, 'kps') and face.kps is not None:
            landmarks = face.kps.astype(np.float32)
            
            # Verify shape
            if landmarks.shape == (5, 2):
                logging.info(f"[FacePreserve][Landmark] Detected {landmarks.shape[0]} points")
                return landmarks
            else:
                logging.warning(f"[FacePreserve][Landmark] Unexpected landmark shape: {landmarks.shape}")
                return None
        else:
            # Fallback: try to extract from landmark_2d_106 if available
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                lm_106 = face.landmark_2d_106
                
                # Extract 5 keypoints from 106 landmarks
                # Left eye center: average of points 36-39
                # Right eye center: average of points 42-45
                # Nose tip: point 30
                # Mouth left: point 48
                # Mouth right: point 54
                
                # Note: InsightFace 106 landmark indices may vary
                # Using common face landmark indices
                left_eye = np.mean([lm_106[36], lm_106[39]], axis=0)
                right_eye = np.mean([lm_106[42], lm_106[45]], axis=0)
                nose = lm_106[30]
                mouth_left = lm_106[48]
                mouth_right = lm_106[54]
                
                landmarks = np.array([
                    left_eye,
                    right_eye,
                    nose,
                    mouth_left,
                    mouth_right
                ], dtype=np.float32)
                
                logging.info(f"[FacePreserve][Landmark] Detected {landmarks.shape[0]} points (from 106)")
                return landmarks
            else:
                logging.warning("[FacePreserve][Landmark] No keypoints found in face object")
                return None
                
    except Exception as e:
        logging.error(f"[FacePreserve][Landmark] Error extracting landmarks: {e}")
        return None


