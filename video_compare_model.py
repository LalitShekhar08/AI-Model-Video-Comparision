import cv2
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet.preprocess_input import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
IMG_SIZE = 224

# --- 1. LOAD MODEL ---
print("Loading AI Model (MobileNetV2)... Please wait.")
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
print("Model Loaded Successfully!")

def process_video_to_embedding(video_path):
    """
    Reads a video file from disk, processes frames, 
    and returns a single embedding vector.
    """
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    # Get total frames (for progress indication)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path} ({total_frames} frames)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Optimization: Process every 10th frame (Faster for long uploaded videos)
        if frame_count % 10 == 0:
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        print("Error: Video seems empty or could not be read.")
        return None

    # Batch processing
    frames_arr = np.array(frames)
    processed_frames = preprocess_input(frames_arr)
    
    # Extract Features
    features = model.predict(processed_frames, verbose=1)
    
    # Average features to get one vector
    video_embedding = np.mean(features, axis=0).reshape(1, -1)
    
    return video_embedding

# --- MAIN INTERFACE ---

print("\n" + "="*40)
print("   AI VIDEO COMPARISON (FILE UPLOAD)   ")
print("="*40)

# 1. Ask for File Paths
# Tip: You can drag and drop the file into the terminal to get the path
ref_path = input("Enter the path/name of the REFERENCE video (e.g. teacher.mp4): ").strip().strip('"')
user_path = input("Enter the path/name of the USER video (e.g. student.mp4): ").strip().strip('"')

# 2. Process
print("\n--- Phase 1: Processing Reference Video ---")
ref_vector = process_video_to_embedding(ref_path)

print("\n--- Phase 2: Processing User Video ---")
user_vector = process_video_to_embedding(user_path)

# 3. Compare
if ref_vector is not None and user_vector is not None:
    similarity = cosine_similarity(ref_vector, user_vector)[0][0]
    score_percent = similarity * 100
    
    print("\n" + "="*30)
    print(f"MATCH SCORE: {score_percent:.2f}%")
    print("="*30)
    
    # Interpretation
    if score_percent > 85:
        print("Verdict: EXCELLENT MATCH! ✅")
    elif score_percent > 65:
        print("Verdict: GOOD ATTEMPT. ⚠️")
    else:
        print("Verdict: NO MATCH. ❌")
else:
    print("\nCould not complete comparison due to file errors.")