#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser
import cv2
import numpy as np
import json
from facefusion import face_detector, face_store, state_manager
from facefusion.typing import VisionFrame


def read_image(image_path: str) -> np.ndarray:
    """Read an image and convert to RGB format"""
    print(f"Reading image from: {os.path.abspath(image_path)}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def prepare_detect_frame(vision_frame: np.ndarray, size: str = '640,640') -> np.ndarray:
    """Prepare frame for detection using facefusion's method"""
    # Don't resize here - let the detector handle it
    detect_frame = (vision_frame.astype(np.float32) - 127.5) / 128.0
    return detect_frame

def get_reference_face(image: np.ndarray) -> dict:
    """Get the first face with highest score from an image"""
    print("Checking face detector...", end=' ')
    
    if not face_detector.pre_check():
        print("FAILED")
        raise ValueError("Failed to initialize face detector!")
    
    print("OK")
    print("Running face detection...")
    try:
        # First convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        print(f"Original image shape: {image.shape}")
        
        # Run detection with original image
        bounding_boxes, scores, landmarks = face_detector.detect_faces(image)
        
        if len(scores) > 0:
            # Filter faces with score > 0.8
            high_confidence_indices = [i for i, score in enumerate(scores) if score > 0.8]
            
            if not high_confidence_indices:
                print("No faces with high confidence found")
                return None
                
            # Among high confidence faces, get the largest one (closest to camera)
            areas = [(bounding_boxes[i][2] - bounding_boxes[i][0]) * 
                    (bounding_boxes[i][3] - bounding_boxes[i][1]) 
                    for i in high_confidence_indices]
            
            best_index = high_confidence_indices[np.argmax(areas)]
            best_score = scores[best_index]
            best_landmarks = landmarks[best_index]
            
            # Calculate and print orientation scores
            face_scores = get_face_scores(best_landmarks)
            print(f"Selected face {best_index+1} with scores:")
            print(f"Detection score: {best_score:.3f}")
            print(f"Horizontal score: {face_scores['horizontal']:.3f}")
            print(f"Vertical score: {face_scores['vertical']:.3f}")
            print(f"Occlusion score: {face_scores['occlusion']:.3f}")
            print(f"Total orientation score: {face_scores['total']:.3f}")
            
            return {
                'bbox': bounding_boxes[best_index],
                'score': best_score,
                'landmarks': best_landmarks
            }
    except Exception as e:
        print(f"Error during face detection: {str(e)}")
        return None
    
    return None

def estimate_vertical_orientation(face_landmark_5):
    """Estimate vertical orientation using 5-point landmarks"""
    nose_y = face_landmark_5[2][1]    # Nose landmark
    mouth_y = (face_landmark_5[3][1] + face_landmark_5[4][1]) / 2  # Average of mouth landmarks

    if mouth_y < nose_y - 10:  # Adjust threshold as needed
        return "face up"
    elif mouth_y > nose_y + 10:  # Adjust threshold as needed
        return "face down"
    else:
        return "neutral"

def check_for_occlusion(face_landmark_5):
    """Calculate occlusion score using feature distances"""
    # Get landmarks
    left_eye = face_landmark_5[0]
    right_eye = face_landmark_5[1]
    nose = face_landmark_5[2]
    left_mouth = face_landmark_5[3]
    right_mouth = face_landmark_5[4]
    
    # Calculate eye distance as reference
    eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + 
                          (right_eye[1] - left_eye[1])**2)
    
    # Calculate feature widths
    lip_width = np.sqrt((right_mouth[0] - left_mouth[0])**2 + 
                       (right_mouth[1] - left_mouth[1])**2)
    
    # Calculate ratios
    lip_ratio = lip_width / eye_distance if eye_distance > 0 else 0
    
    # Calculate occlusion penalties (0-0.33 range for each feature)
    mouth_penalty = 0.33 * (1 - min(lip_ratio / 0.9, 1.0))
    eye_penalty = 0.33 * (1 - min(eye_distance / (lip_width * 1.2), 1.0))
    nose_penalty = 0.33 * (1 - min(abs(nose[1] - (left_eye[1] + right_eye[1])/2) / (eye_distance * 0.5), 1.0))
    
    print(f"DEBUG - Penalties: Eyes={eye_penalty:.2f}, Nose={nose_penalty:.2f}, Mouth={mouth_penalty:.2f}")
    
    # Total penalty (sum of all three)
    total_penalty = mouth_penalty + eye_penalty + nose_penalty
    
    # Final score (1 minus total penalty)
    final_score = max(0, 1.0 - total_penalty)
    
    print(f"DEBUG - Total penalty: {total_penalty:.2f}, Final score: {final_score:.2f}")
    
    return final_score

def calculate_horizontal_score(face_landmarks):
    face_angle = estimate_face_angle(face_landmarks)
    # Create a continuous score that decreases as angle increases
    if face_angle <= 30:
        return 1.0
    elif face_angle >= 90:
        return 0.0
    else:
        # Linear interpolation between 30° (1.0) and 90° (0.0)
        return 1.0 - ((face_angle - 30) / 60)

def calculate_vertical_score(face_landmarks):
    vertical_orientation = estimate_vertical_orientation(face_landmarks)
    nose_y = face_landmarks[2][1]    # Nose landmark
    mouth_y = (face_landmarks[3][1] + face_landmarks[4][1]) / 2  # Average of mouth landmarks
    
    # Calculate vertical angle using relative positions
    vertical_diff = abs(mouth_y - nose_y) / (face_landmarks[1][0] - face_landmarks[0][0])  # Normalized by eye distance
    
    # Convert to score (smaller difference = higher score)
    if vertical_diff <= 0.3:  # Normal range
        return 1.0
    elif vertical_diff >= 0.8:  # Extreme angle
        return 0.0
    else:
        # Linear interpolation between 0.3 (1.0) and 0.8 (0.0)
        return 1.0 - ((vertical_diff - 0.3) / 0.5)

def calculate_occlusion_score(face_landmarks):
    score = check_for_occlusion(face_landmarks)
    print(f"DEBUG - Final occlusion score: {score:.3f}")
    return score

def estimate_face_angle(landmarks):
    """Estimate horizontal face angle using eye positions with 5-point landmarks"""
    # YOLOFace landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    left_eye = landmarks[0]   # Left eye
    right_eye = landmarks[1]  # Right eye
    
    # Calculate eye distance and angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    # Calculate angle in degrees
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    return angle

def get_face_scores(landmarks):
    """Calculate combined face orientation scores"""
    horizontal_score = calculate_horizontal_score(landmarks)
    vertical_score = calculate_vertical_score(landmarks)
    occlusion_score = calculate_occlusion_score(landmarks)
    
    # Combined score (you can adjust weights if needed)
    total_score = (horizontal_score * 0.4 + 
                  vertical_score * 0.4 + 
                  occlusion_score * 0.2)
    
    return {
        'horizontal': horizontal_score,
        'vertical': vertical_score,
        'occlusion': occlusion_score,
        'total': total_score
    }

def filter_video(video_path: str, output_path: str, reference_face: dict,
                horizontal_threshold: float = 0.6,
                vertical_threshold: float = 0.6,
                occlusion_threshold: float = 0.6):
    """Filter frames in a video that don't match the reference face"""
    # Create temp directory
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Created temp directory: {temp_dir}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frames needed for 5 seconds
    min_segment_frames = int(fps * 5)
    chunk_size = min_segment_frames * 4  # Process 20 seconds at a time
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video in chunks
    frame_idx = 0
    chunk_frames = []
    chunk_valid = []
    segments = []
    chunk_start = 0
    
    # Save metadata for potential resume
    metadata = {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'min_segment_frames': min_segment_frames
    }
    with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    print("\nProcessing video in chunks...")
    
    # Define score threshold
    SCORE_THRESHOLD = 0.6
    
    # These state settings are crucial and must be set BEFORE any face detection
    state_manager.set_item('face_detector_score', horizontal_threshold)
    state_manager.set_item('face_landmarker_score', vertical_threshold)
    state_manager.set_item('download_providers', ['huggingface'])
    state_manager.set_item('face_detector_model', 'yoloface')
    state_manager.set_item('face_detector_size', '640x640')
    state_manager.set_item('execution_providers', ['cpu'])
    state_manager.set_item('execution_thread_count', 1)
    state_manager.set_item('execution_queue_count', 1)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if chunk_frames:
                process_chunk(chunk_frames, chunk_valid, segments, chunk_start, min_segment_frames)
            break
            
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{total_frames}", end='\r')
        
        # Save frame to temp directory
        frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Store frame path instead of frame
        chunk_frames.append(frame_path)
        
        # Detect faces and calculate scores
        faces_data = get_reference_face(frame)
        is_valid_frame = False
        
        if faces_data is not None:
            scores = get_face_scores(faces_data['landmarks'])
            
            # Check individual thresholds
            is_valid_frame = (
                scores['horizontal'] >= horizontal_threshold and
                scores['vertical'] >= vertical_threshold and
                scores['occlusion'] >= occlusion_threshold
            )
            
            if frame_idx % 100 == 0:  # Print scores every 100 frames
                print(f"\nFrame {frame_idx} scores:")
                print(f"Horizontal: {scores['horizontal']:.2f} (threshold: {horizontal_threshold})")
                print(f"Vertical: {scores['vertical']:.2f} (threshold: {vertical_threshold})")
                print(f"Occlusion: {scores['occlusion']:.2f} (threshold: {occlusion_threshold})")
        
        chunk_valid.append(is_valid_frame)
        
        # Save progress
        if frame_idx % 100 == 0:  # Save every 100 frames
            progress = {
                'frame_idx': frame_idx,
                'segments': segments,
                'chunk_start': chunk_start
            }
            with open(os.path.join(temp_dir, 'progress.json'), 'w') as f:
                json.dump(progress, f)
        
        # Process chunk when it's full
        if len(chunk_frames) >= chunk_size:
            process_chunk(chunk_frames, chunk_valid, segments, chunk_start, min_segment_frames)
            chunk_start += len(chunk_frames)
            chunk_frames = []
            chunk_valid = []
    
    # Write valid segments
    print(f"\nWriting {len(segments)} segments...")
    total_kept_frames = 0
    
    for segment_start, segment_end in segments:
        # Write segment frames from temp files
        for frame_num in range(segment_start, segment_end):
            frame_path = os.path.join(temp_dir, f"frame_{frame_num+1:06d}.jpg")
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)
                    total_kept_frames += 1
                    print(f"Writing frame {total_kept_frames}", end='\r')
    
    # Clean up
    cap.release()
    out.release()
    
    # Print summary
    total_segments = len(segments)
    total_duration = total_kept_frames / fps
    print(f"\nSummary:")
    print(f"Found {total_segments} segments of 10+ seconds")
    print(f"Total frames kept: {total_kept_frames}/{total_frames}")
    print(f"Output video duration: {total_duration:.1f} seconds")
    
    # Print segment details
    print("\nSegment details:")
    for i, (start, end) in enumerate(segments, 1):
        duration = (end - start) / fps
        print(f"Segment {i}: {duration:.1f} seconds ({start} to {end} frames)")
    
    # Ask user if they want to keep temp files
    keep_temp = input("\nKeep temporary files? (y/n): ").lower().strip() == 'y'
    if not keep_temp:
        print("Cleaning up temporary files...")
        import shutil
        shutil.rmtree(temp_dir)
    else:
        print(f"Temporary files kept in: {temp_dir}")

def process_chunk(chunk_frames, chunk_valid, segments, chunk_start, min_segment_frames):
    """Process a chunk of frames to find valid segments"""
    start = None
    current_segment_length = 0
    
    for i in range(len(chunk_valid)):
        if chunk_valid[i]:  # Valid frame
            if start is None:
                start = i
            current_segment_length += 1
        else:  # Invalid frame
            if start is not None:
                # If segment is long enough, keep it
                if current_segment_length >= min_segment_frames:
                    segments.append((chunk_start + start, chunk_start + i))
                start = None
                current_segment_length = 0
    
    # Handle the last segment
    if start is not None and current_segment_length >= min_segment_frames:
        segments.append((chunk_start + start, chunk_start + len(chunk_valid)))

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--reference', help='Path to reference image', required=True)
    parser.add_argument('--target', help='Path to target video', required=True)
    parser.add_argument('--output', help='Path to output video', required=True)
    parser.add_argument('--face-detector-score', type=float, default=0.5,
                        help='Face detector score threshold')
    parser.add_argument('--face-landmarker-score', type=float, default=0.5,
                        help='Face landmarker score threshold')
    parser.add_argument('--horizontal-threshold', type=float, default=0.4,
                        help='Horizontal orientation threshold')
    parser.add_argument('--vertical-threshold', type=float, default=0.4,
                        help='Vertical orientation threshold')
    parser.add_argument('--occlusion-threshold', type=float, default=0.4,
                        help='Occlusion threshold')
    args = parser.parse_args()

    # These state settings are crucial and must be set BEFORE any face detection
    state_manager.set_item('face_detector_score', args.face_detector_score)
    state_manager.set_item('face_landmarker_score', args.face_landmarker_score)
    state_manager.set_item('download_providers', ['huggingface'])
    state_manager.set_item('face_detector_model', 'yoloface')
    state_manager.set_item('face_detector_size', '640x640')
    state_manager.set_item('execution_providers', ['cpu'])
    state_manager.set_item('execution_thread_count', 1)
    state_manager.set_item('execution_queue_count', 1)

    # Force model download/check before proceeding
    if not face_detector.pre_check():
        print('Failed to initialize face detector!')
        sys.exit(1)

    # Then load reference face
    print('Loading reference face...')
    reference_frame = read_image(args.reference)
    reference_face = get_reference_face(reference_frame)

    if reference_face is None or reference_face['score'] < args.face_detector_score:
        print('No face detected in reference image or score too low!')
        sys.exit(1)

    print(f"Reference face detected with score: {reference_face['score']:.2f}")

    # Process video
    print('Processing video...')
    try:
        filter_video(args.target, args.output, reference_face, 
                    horizontal_threshold=args.horizontal_threshold,
                    vertical_threshold=args.vertical_threshold,
                    occlusion_threshold=args.occlusion_threshold)
        print(f'Filtered video saved to: {args.output}')
    except Exception as e:
        print(f'Error processing video: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main()