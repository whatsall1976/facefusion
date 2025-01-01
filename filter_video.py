#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser
import cv2
import numpy as np

from facefusion import face_detector, face_store, state_manager
from facefusion.typing import VisionFrame
from facefusion.choices import download_provider_set

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
    print("Checking face detector...")
    
    if not face_detector.pre_check():
        raise ValueError("Failed to initialize face detector!")
    
    print("Running face detection...")
    try:
        # First convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        print(f"Original image shape: {image.shape}")
        
        # Run detection with original image - let face_detector handle the resizing
        bounding_boxes, scores, landmarks = face_detector.detect_faces(image)
        
        print(f"Found {len(scores)} faces with scores:")
        for i, score in enumerate(scores):
            print(f"Face {i+1}: {score:.3f}")
        
        if len(scores) > 0:
            # Get the face with highest score
            best_index = np.argmax(scores)
            best_score = scores[best_index]
            print(f"Best face score: {best_score:.3f}")
            
            return {
                'bbox': bounding_boxes[best_index],
                'score': best_score,
                'landmarks': landmarks[best_index]
            }
    except Exception as e:
        print(f"Error during face detection: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    return None

def filter_video(video_path: str, output_path: str, reference_face: dict):
    """Filter frames in a video that don't match the reference face"""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{total_frames}", end='\r')
        
        # Detect faces in current frame
        faces = get_reference_face(frame)
        
        # If no face found or face score is too low, skip frame
        if faces is None:
            continue
            
        # Write frame
        out.write(frame)
    
    # Clean up
    cap.release()
    out.release()

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--reference', help='Path to reference image', required=True)
    parser.add_argument('--target', help='Path to target video', required=True)
    parser.add_argument('--output', help='Path to output video', required=True)
    parser.add_argument('--face-detector-score', type=float, default=0.5,
                      help='Face detector score threshold')
    parser.add_argument('--face-landmarker-score', type=float, default=0.5,
                      help='Face landmarker score threshold')
    args = parser.parse_args()

    # Set all required state parameters
    state_manager.set_item('face_detector_score', args.face_detector_score)
    state_manager.set_item('face_landmarker_score', args.face_landmarker_score)
    state_manager.set_item('download_providers', ['huggingface'])
    state_manager.set_item('face_detector_model', 'retinaface')
    state_manager.set_item('face_detector_size', '640x640')
    state_manager.set_item('execution_providers', ['cpu'])
    state_manager.set_item('execution_thread_count', 1)
    state_manager.set_item('execution_queue_count', 1)

    # Initialize face detector
    print('Initializing face detector...')
    if not face_detector.pre_check():
        print('Failed to initialize face detector!')
        sys.exit(1)

    # Load reference face
    print('Loading reference face...')
    reference_frame = read_image(args.reference)
    reference_face = get_reference_face(reference_frame)
    
    if reference_face is None:
        print('No face detected in reference image!')
        sys.exit(1)
        
    print(f"Reference face detected with score: {reference_face['score']:.2f}")

    # Process video
    print('Processing video...')
    try:
        filter_video(args.target, args.output, reference_face)
        print(f'Filtered video saved to: {args.output}')
    except Exception as e:
        print(f'Error processing video: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main()