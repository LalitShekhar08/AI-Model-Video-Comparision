import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Pose Matcher",)

# --- 1. SETUP MEDIAPIPE AI ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_pose_embedding(video_file):
    """
    Extracts the SKELETON (33 Landmarks) from the video.
    This is 'Real AI' - it ignores background and lighting.
    """
    # Fix for Windows: Write to temp file, then CLOSE it.
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close() 
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Initialize the AI Model
    # model_complexity=1 is a balance between speed and accuracy
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
    
    all_landmarks = []
    
    # UI Progress
    status_text = st.empty()
    status_text.text("⏳ AI is scanning movement...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR (OpenCV) to RGB (AI needs RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with AI
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extract 33 points (x, y, z for nose, shoulders, hips, etc.)
            frame_data = []
            for lm in results.pose_landmarks.landmark:
                # We flatten the x, y, z coordinates into a long list
                frame_data.extend([lm.x, lm.y, lm.z])
            all_landmarks.append(frame_data)
            
    cap.release()
    try:
        os.remove(tfile.name)
    except:
        pass
    
    status_text.empty()
    
    if not all_landmarks:
        return None
        
    # Average the movement over time to get a "Motion Signature"
    video_embedding = np.mean(all_landmarks, axis=0).reshape(1, -1)
    
    return video_embedding

# --- APP INTERFACE ---

st.title("AI Model Video Comparision")
st.markdown("Model Developed by **Google MediaPipe**  Compares **Movments**, not just pixels.")

col1, col2 = st.columns(2)

with col1:
    st.header("Video1")
    ref_file = st.file_uploader("Upload Reference", type=['mp4', 'mov'], key='ref')

with col2:
    st.header("Video2")
    user_file = st.file_uploader("Upload Attempt", type=['mp4', 'mov'], key='user')

if st.button("Analyze Movement"):
    if ref_file and user_file:
        with st.spinner('Extracting Structure...'):
            
            # 1. Get AI Embeddings
            vec1 = get_pose_embedding(ref_file)
            vec2 = get_pose_embedding(user_file)
            
            if vec1 is not None and vec2 is not None:
                # 2. Compare using Math (Cosine Similarity)
                similarity = cosine_similarity(vec1, vec2)[0][0]
                score = similarity * 100
                
                st.divider()
                st.metric("Movement Similarity", f"{score:.1f}%")
                
                # AI Interpretation
                if score > 95:
                    st.success("Perfect Match! 🥇 Your form is identical.")
                    st.balloons()
                elif score > 85:
                    st.info("Great Form! 🥈 Very slight differences.")
                elif score > 70:
                    st.warning("Good Attempt. 🥉 Watch your limb positioning.")
                else:
                    st.error("Different Movement Detected. ❌ Try again.")
                    
            else:
                st.error("Error: The AI could not find a human in one of the videos.")
    else:
        st.info("Please upload both videos to start.")