# Face Mask Detector using CNN

## Overview
This project implements a Deep Learning solution to identify whether individuals are wearing face masks. Built with TensorFlow and Keras, the system is designed for real-world applications such as workplace compliance, public safety monitoring, and security surveillance.

## Features
* **Dual Detection Modes:** Supports both high-precision static image detection and real-time webcam processing.
* **CNN Architecture:** Optimized Convolutional Neural Network for binary classification (Mask vs. No Mask).
* **Computer Vision Integration:** Uses OpenCV for face localization and frame-by-frame video processing.
* **Robust Evaluation:** Includes performance metrics and visualization using Matplotlib and Scikit-learn.

## Tech Stack
* **Deep Learning:** TensorFlow 2.16.1, Keras
* **Computer Vision:** OpenCV
* **Data Science:** NumPy, Scikit-learn, Matplotlib

## Quick Start
1. **Clone & Setup:**
   ```bash
   git clone [https://github.com/yourusername/face-mask-detection.git](https://github.com/yourusername/face-mask-detection.git)
   cd face-mask-detection
   python -m venv venv
   source venv/bin/activate  # .\venv\Scripts\activate for Windows
   pip install -r requirements.txt

**2. Run Image Detection:** python src/image_predict.py

**3. Run Real-Time Detection**: python src/webcam_predict.py
