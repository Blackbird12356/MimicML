# MimicML
MimicML - From TikTok Inspiration to a Functional ML Pipeline!
"Good evening, everyone! This weekend, I decided to challenge myself by recreating a project I saw on TikTok. While it started as a 'just for fun' experiment, it quickly turned into a deep dive into ML Engineering with Python.
I wanted to build a system that doesn't just recognize gestures, but actually learns from the user in real-time.
The Technical Stack & Resources:
MediaPipe (Google): I chose this for high-fidelity hand tracking. It allowed me to extract 21 3D hand landmarks, transforming raw pixels into lightweight coordinate data.
Scikit-learn: Used to implement the brain of the project. I built a classification model that enables on-device training, allowing the system to learn new gestures in seconds without needing a massive pre-trained dataset.
OpenCV: The backbone for real-time video processing, frame manipulation, and managing the camera stream.
Python: The core language used to orchestrate the entire pipeline—from data collection (capturing 30 frames per gesture) to real-time inference and GIF triggering.
The Engineering Challenge:
The most interesting part was creating the data collection workflow. I engineered a way for the model to 'study' user behavior by taking 30 samples in 10 seconds, then immediately mapping those patterns to specific outputs (GIFs).
This project reinforced my passion for ML Engineering—specifically how to make AI interactive, fast, and user-adaptive.
