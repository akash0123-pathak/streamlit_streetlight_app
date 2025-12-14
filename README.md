# Smart Streetlight AI – Streamlit Application

An AI-powered Smart Streetlight system that predicts whether a streetlight should be ON or OFF
based on traffic density, visibility conditions, brightness, and time-based features.

This project is built using open-source technologies only and does NOT require any API keys,
tokens, or secrets.

---

## Project Overview

Cities waste energy by keeping streetlights ON even when they are not needed.
This project demonstrates how computer vision and machine learning can intelligently
control streetlights using real-world conditions.

The application:
- Extracts features from video frames
- Estimates fog / visibility
- Trains multiple machine learning models
- Performs real-time inference
- Displays predictions using Streamlit

---

## How the System Works

### Feature Extraction
From each video frame:
- Vehicle count
- Pedestrian count
- Brightness
- Contrast
- Fog score
- Hour and day information
- Weather (synthetic)

Detection:
- YOLOv8 if available
- Background subtraction fallback

---

### Dataset Generation
A syn
