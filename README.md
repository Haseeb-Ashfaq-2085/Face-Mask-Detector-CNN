# Face-Mask-Detector-using-CNN
Face Mask Detection using CNN | Computer Vision Project

This project is built using Deep Learning and CNN (Convolutional Neural Networks) with TensorFlow and Keras to detect whether a person is wearing a face mask or not. The system supports both real-time webcam detection and image-based detection with a trained model for high accuracy. It leverages OpenCV, NumPy, Matplotlib, and Scikit-learn for image processing, visualization, and model evaluation.

This project demonstrates how computer vision can be applied in real-world scenarios like pandemic safety monitoring, smart surveillance, and workplace compliance. The trained CNN model achieves excellent performance and can even be integrated into CCTV systems to detect robbers hiding their faces.

How to Run the Project
git clone https://github.com/<your-username>/face-mask-detection.git
cd face-mask-detection
python -m venv venv
.\venv\Scripts\activate    # For Windows
pip install -r requirements.txt
python src/image_predict.py  # For image-based detection
python src/webcam_predict.py # For real-time webcam detection


Requirements:
tensorflow==2.16.1, opencv-python, numpy, matplotlib, scikit-learn
