from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow import keras
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# FIX #1: Use best_fer_model instead of fer_model_final
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_fer_model.keras')
model = keras.models.load_model(MODEL_PATH)

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48

# Face cascade for face detection
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def detect_and_crop_face(img_array):
    """Detect face and crop it."""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        # No face detected, return resized original
        return cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    # Use the largest face
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    
    # Add padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(gray.shape[1] - x, w + padding * 2)
    h = min(gray.shape[0] - y, h + padding * 2)
    
    face = gray[y:y+h, x:x+w]
    
    return face

def preprocess_image(img_array):
    """Enhance image for better detection."""
    # Detect and crop face
    face = detect_and_crop_face(img_array)
    
    # Apply histogram equalization (improves contrast)
    face = cv2.equalizeHist(face)
    
    # Apply Gaussian blur to reduce noise
    face = cv2.GaussianBlur(face, (5, 5), 0)
    
    # Resize to model input size
    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    face_normalized = face_resized.astype(np.float32) / 255.0
    
    return face_normalized

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict emotion from base64 image."""
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Preprocess image
        img_input = preprocess_image(img_array)
        img_input = img_input.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        # Predict
        prediction = model.predict(img_input, verbose=0)
        emotion_id = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]) * 100)
        
        return jsonify({
            'emotion': EMOTION_LABELS[emotion_id],
            'confidence': confidence,
            'probabilities': {
                EMOTION_LABELS[i]: float(prediction[0][i] * 100) 
                for i in range(len(EMOTION_LABELS))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK', 'model': 'FER v1.0'})

@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    return jsonify({'emotions': EMOTION_LABELS})

@app.route('/api/debug', methods=['POST'])
def debug():
    """Debug endpoint to check image processing."""
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        
        img_array = np.array(img)
        
        print(f"Original shape: {img_array.shape}")
        print(f"Original dtype: {img_array.dtype}")
        print(f"Original min/max: {img_array.min()}/{img_array.max()}")
        
        # After preprocessing (now uses the defined function)
        face = detect_and_crop_face(img_array)
        face = cv2.equalizeHist(face)
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        print(f"Processed shape: {face_normalized.shape}")
        print(f"Processed dtype: {face_normalized.dtype}")
        print(f"Processed min/max: {face_normalized.min()}/{face_normalized.max()}")
        
        return jsonify({
            'original_shape': str(img_array.shape),
            'original_dtype': str(img_array.dtype),
            'original_min_max': f"{img_array.min()}/{img_array.max()}",
            'processed_shape': str(face_normalized.shape),
            'processed_dtype': str(face_normalized.dtype),
            'processed_min_max': f"{face_normalized.min()}/{face_normalized.max()}",
            'status': 'Debug info logged'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)