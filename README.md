# FER Emotion Detector

> Real-time Facial Emotion Recognition powered by a Keras CNN, Flask REST API, and a React + Vite frontend.

---

## Demo

|    Webcam Feed     |                        Prediction                         |
| :----------------: | :-------------------------------------------------------: |
| Live camera stream | Angry / Disgust / Fear / Happy / Sad / Surprise / Neutral |

---

## Project Structure

```
fer-emotion-detector/
├── backend/               # Flask API + Keras model
│   ├── app.py
│   └── models/
│       └── best_fer_model.keras
├── frontend/              # React + Vite web app
│   └── src/
│       ├── App.jsx
│       └── App.css
├── training/              # Model training scripts & notebook
│   ├── FER_Training_Colab.ipynb
│   └── train.py
├── dataset/               # FER2013 data (not committed — see below)
│   ├── fer2013.csv
│   ├── train/
│   └── test/
└── requirements.txt
```

---

## Dataset

This project uses the **FER-2013** dataset.

| Detail  | Info                                                                             |
| ------- | -------------------------------------------------------------------------------- |
| Source  | [Kaggle — FER2013 by msambare](https://www.kaggle.com/datasets/msambare/fer2013) |
| Images  | 35,887 grayscale 48×48 px images                                                 |
| Classes | 7 emotions                                                                       |
| Split   | 28,709 train / 7,178 test                                                        |

**Download instructions:**

1. Go to [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Download and unzip into `dataset/`
3. Your folder should contain `train/` and `test/` subdirectories each with 7 emotion folders.

> The dataset is excluded from this repository due to its size (~63 MB).

---

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Webcam

### 1. Backend (Flask + Keras)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the API server
cd backend
python app.py
```

The API runs at `http://localhost:5000`.

**Endpoints:**

- `GET /api/health` — Health check
- `POST /api/predict` — Send base64 image, receive emotion prediction

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser, allow camera access, and click **Start Detection**.

---

## How It Works

```
Webcam frame
    │
    ▼
[Frontend] Canvas captures frame as base64 PNG
    │
    ▼  POST /api/predict
[Backend] Decode image
    │
    ▼
Haar Cascade face detection
    │
    ▼
Histogram equalization + Gaussian blur
    │
    ▼
Resize to 48x48 + Normalize
    │
    ▼
Keras CNN model inference
    │
    ▼
JSON response (emotion, confidence)
    │
    ▼
[Frontend] Display result
```

---

## Training Your Own Model

Open the training notebook in Google Colab for GPU-accelerated training:

```
training/FER_Training_Colab.ipynb
```

Or train locally:

```bash
cd training
python train.py
```

Place the output `.keras` file in `backend/models/` and update the model path in `backend/app.py`.

---

## Tech Stack

| Layer            | Technology             |
| ---------------- | ---------------------- |
| Model            | Keras / TensorFlow CNN |
| Face Detection   | OpenCV Haar Cascade    |
| Backend          | Flask, Flask-CORS      |
| Frontend         | React 19, Vite 7       |
| Image Processing | Pillow, NumPy, OpenCV  |

---

## Requirements

Key Python packages from `requirements.txt`:

```
tensorflow
keras
opencv-python
flask
flask-cors
Pillow
numpy
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

<div align="center">
  FER2013 dataset by <a href="https://www.kaggle.com/msambare">msambare</a> on Kaggle
</div>
