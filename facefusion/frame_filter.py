from typing import List, Optional, Tuple
import cv2
import numpy as np

from facefusion.typing import VisionFrame, Face, FaceSet, BoundingBox, Score, FaceLandmark5
from facefusion import face_detector, face_landmarker, face_store, state_manager

def filter_frames(
    frames: List[VisionFrame],
    reference_faces: Optional[FaceSet] = None
) -> List[VisionFrame]:
    """
    Filter frames to keep only those containing good quality target faces.
    Uses the same detection and scoring logic as the main face fusion pipeline.
    
    Args:
        frames: List of input frames to filter
        reference_faces: Optional reference faces to match against
        
    Returns:
        List of frames containing detected target faces with good quality scores
    """
    filtered_frames = []
    
    for frame in frames:
        # Get face detections using the configured detector model
        bounding_boxes, face_scores, landmarks_5 = face_detector.detect_faces(frame)
        
        # Skip frame if no faces detected
        if not bounding_boxes:
            continue
            
        # Check each detected face
        valid_faces = []
        for bbox, face_score, landmark_5 in zip(bounding_boxes, face_scores, landmarks_5):
            # Skip if face detection score is too low
            if face_score < state_manager.get_item('face_detector_score'):
                continue
                
            # Get face angle for landmark detection
            face_angle = 0 # You might want to calculate this based on landmarks
                
            # Get 68-point landmarks and score
            landmarks_68, landmark_score = face_landmarker.detect_face_landmarks(
                frame, 
                bbox,
                face_angle
            )
            
            # Skip if landmark detection score is too low
            if landmark_score < state_manager.get_item('face_landmarker_score'):
                continue
                
            valid_faces.append({
                'bbox': bbox,
                'score': face_score,
                'landmark_5': landmark_5,
                'landmark_68': landmarks_68,
                'landmark_score': landmark_score
            })
        
        # Skip frame if no valid faces
        if not valid_faces:
            continue
            
        # If reference faces provided, check for matches
        if reference_faces:
            has_target_face = False
            for face in valid_faces:
                # Compare against reference faces
                for ref_faces in reference_faces.values():
                    if _is_similar_face(face, ref_faces):
                        has_target_face = True
                        break
                if has_target_face:
                    break
                    
            if not has_target_face:
                continue
        
        # Keep frame if it passes all quality checks
        filtered_frames.append(frame)
    
    return filtered_frames

def _is_similar_face(face: Face, reference_faces: List[Face]) -> bool:
    """
    Check if face matches any of the reference faces using the project's
    similarity thresholds and scoring system.
    """
    for ref_face in reference_faces:
        # Get reference scores
        ref_score = ref_face.get('score', 0)
        ref_landmark_score = ref_face.get('landmark_score', 0)
        
        # Compare scores
        if (face['score'] >= ref_score * 0.9 and 
            face['landmark_score'] >= ref_landmark_score * 0.9):
            return True
    return False

def save_filtered_video(
    input_path: str,
    output_path: str,
    reference_faces: Optional[FaceSet] = None
) -> None:
    """
    Load video, filter frames with poor quality faces, and save filtered video.
    """
    # Open video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # Filter frames
    filtered_frames = filter_frames(
        frames,
        reference_faces
    )
    
    # Write filtered frames
    for frame in filtered_frames:
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()