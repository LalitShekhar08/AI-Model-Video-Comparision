AI Pose Comparison & Feedback System
📌 Overview
This project is an AI-powered application that compares human body movements in two videos (a user video vs. a reference video) to assess form and accuracy. It uses Google MediaPipe for skeletal tracking and Cosine Similarity to mathematically grade how closely the user matches the reference pose.

The application is built with Streamlit, allowing for a simple, interactive web-based interface.

🚀 Features
Pose Estimation: Detects 33 skeletal landmarks (joints) on the human body using MediaPipe BlazePose.

Mathematical Scoring: Calculates a similarity score (0-100%) using Cosine Similarity to quantify how well the user mimics the reference.

Visual Feedback: Overlays skeletal landmarks on the video feed for visual analysis.

Privacy-First: All processing is done locally on the CPU; no video data is sent to external servers.

🛠️ Tech Stack
Language: Python 3.11 (Recommended for stability)

Interface: Streamlit

Computer Vision: OpenCV (cv2), MediaPipe

Data/Math: NumPy, Scikit-learn (sklearn)

⚙️ Installation & Setup
Prerequisites
Python 3.11 installed on your machine.

(Optional but recommended) VS Code.

Step 1: Clone or Download
Download this project folder to your local machine.

Step 2: Set up the Environment (Windows)
Note: This project relies on specific library versions. It is highly recommended to use a virtual environment.

Open your terminal inside the project folder.

Create a virtual environment using Python 3.11:

Bash

py -3.11 -m venv venv
Activate the environment:

Bash

.\venv\Scripts\activate
(You should see (venv) appear in your terminal).

Step 3: Install Dependencies
Run the following command to install the required ML libraries:

Bash

pip install mediapipe opencv-python streamlit scikit-learn numpy
▶️ How to Run
Ensure your virtual environment is active (see above).

Run the Streamlit app:

Bash

python -m streamlit run app.py
The app should automatically open in your browser at http://localhost:8501.

🧠 How It Works
Input: The app accepts a video file (Reference) and a second video file (User).

Extraction: MediaPipe scans every frame and extracts the (x, y, z) coordinates for 33 body landmarks (shoulders, elbows, knees, etc.).

Normalization: The data is normalized to account for different camera distances (e.g., standing closer vs. further away).

Comparison: The system calculates the Cosine Similarity between the user's vector and the reference vector.

1.0 (100%): Perfect alignment.

< 0.9: Noticeable deviation in form.

📂 Project Structure
├── app.py                # Main application logic (Streamlit + MediaPipe)
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
└── venv/                 # Virtual environment folder (do not commit to GitHub)
⚠️ Troubleshooting
Error: ModuleNotFoundError: No module named 'mediapipe'

Cause: You are likely running the global Python (3.13) instead of the project environment (3.11).

Fix: Ensure you see (venv) in your terminal. If not, activate it again: .\venv\Scripts\activate.

Error: AttributeError: module 'streamlit' has no attribute '...

Fix: Ensure you don't have a file named streamlit.py in your folder, as it conflicts with the library.
