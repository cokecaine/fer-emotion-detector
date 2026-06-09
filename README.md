# FER Emotion Detector

> Real-time Facial Emotion Recognition powered by a Keras CNN, Flask REST API, and a React + Vite frontend.

---

## Tech Stack

| Layer            | Technology                        |
| ---------------- | --------------------------------- |
| Model            | Keras 3.13 / TensorFlow 2.21 CNN  |
| Face Detection   | OpenCV 4.13 Haar Cascade + MTCNN  |
| Backend          | Flask 3.1, Flask-CORS 6.0         |
| Frontend         | React 19, Vite 7                  |
| Image Processing | Pillow 12, NumPy 2.4, OpenCV 4.13 |
| Runtime          | Python 3.12.3 (WSL), Node.js 18+  |

---

## Project Structure

```
fer-emotion-detector/
├── backend/               # Flask API + Keras model
│   ├── app.py
│   └── models/
│       └── best_fer_model.keras
├── frontend/              # React + Vite web app
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
├── training/              # Model training scripts & notebook
│   ├── FER_Training_Colab.ipynb
│   └── train.py
├── dataset/               # FER2013 data (not committed — see below)
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
| Classes | 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)                 |
| Split   | 28,709 train / 7,178 test                                                        |

**Download instructions:**

1. Go to <https://www.kaggle.com/datasets/msambare/fer2013>
2. Download and unzip into `dataset/`
3. Your folder should contain `train/` and `test/` subdirectories, each with 7 emotion sub-folders.

> The dataset is excluded from this repository due to its size (~63 MB).

---

## Quick Start (WSL)

> **Environment:** WSL (Ubuntu recommended), Python **3.12.3**, Node.js **18+**

### 1. Clone the repo

```bash
git clone https://github.com/cokecaine/fer-emotion-detector.git
cd fer-emotion-detector
```

### 2. Backend — Flask + Keras

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all pinned dependencies
pip install -r requirements.txt

# Start the API server
cd backend
python3 app.py
```

The API will be available at `http://localhost:5000`.

> **WSL tip:** If you need to access the server from your Windows browser, use  
> `http://<WSL-IP>:5000` or enable `localhost` forwarding in your WSL distro settings.

#### API Endpoints

| Method | Path           | Description                                    |
| ------ | -------------- | ---------------------------------------------- |
| GET    | `/api/health`  | Health check — returns `{ "status": "ok" }`   |
| POST   | `/api/predict` | Send base64 image, receive emotion prediction  |

**Example request body:**
```json
{
  "image": "data:image/png;base64,<base64-string>"
}
```

**Example response:**
```json
{
  "emotion": "happy",
  "confidence": 94.3,
  "probabilities": {
    "angry": 0.1,
    "disgust": 0.2,
    "fear": 0.3,
    "happy": 94.3,
    "sad": 1.2,
    "surprise": 3.5,
    "neutral": 0.4
  }
}
```

### 3. Frontend — React + Vite

Open a **new terminal** (WSL or Windows):

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
[Frontend] Canvas captures frame → base64 PNG
    │
    ▼  POST /api/predict
[Backend] Decode base64 image
    │
    ▼
OpenCV Haar Cascade face detection
    │
    ▼
Histogram equalization + Gaussian blur (pre-processing)
    │
    ▼
Resize to 48×48 px + normalize [0, 1]
    │
    ▼
Keras CNN model inference (TensorFlow 2.21)
    │
    ▼
JSON response → { emotion, confidence, probabilities }
    │
    ▼
[Frontend] Display live results with confidence bars
```

---

## Python Dependencies

All packages are pinned in `requirements.txt`. Key ones:

| Package              | Version   | Purpose                          |
| -------------------- | --------- | -------------------------------- |
| `tensorflow`         | 2.21.0    | Deep learning runtime            |
| `keras`              | 3.13.2    | High-level model API             |
| `opencv-python`      | 4.13.0.92 | Face detection & image processing|
| `flask`              | 3.1.3     | REST API server                  |
| `flask-cors`         | 6.0.2     | Cross-origin requests            |
| `pillow`             | 12.1.1    | Image loading & conversion       |
| `numpy`              | 2.4.3     | Array operations                 |
| `mtcnn`              | 1.0.0     | Alternative face detection       |
| `retina-face`        | 0.0.18    | Deep face detection              |
| `gunicorn`           | 25.1.0    | Production WSGI server           |
| `python-dotenv`      | 1.2.2     | Environment variable management  |
| `gdown`              | 5.2.1     | Download model from Google Drive |

> CUDA packages (`nvidia-*`) are included for GPU acceleration on WSL2 with NVIDIA drivers.  
> They are safely ignored on CPU-only environments.

---

## Training Your Own Model

Open the training notebook in Google Colab for GPU-accelerated training:

```
training/FER_Training_Colab.ipynb
```

Or train locally on WSL:

```bash
# Make sure your venv is active
source .venv/bin/activate

cd training
python3 train.py
```

Place the output `.keras` file in `backend/models/` and update the model path in `backend/app.py` if needed.

---

## Troubleshooting

### Camera not working in WSL browser

WSL does not have direct access to hardware. Run the **frontend** from your Windows side (native terminal / VS Code) so the browser can access the webcam normally.

### `ModuleNotFoundError` on startup

Make sure your virtual environment is activated:

```bash
source .venv/bin/activate
```

### CORS error in browser console

The backend must be running **before** you open the frontend. Confirm `http://localhost:5000/api/health` returns a response.

### TensorFlow / CUDA warnings on CPU

These are informational only and won't affect predictions. To suppress them:

```bash
export TF_CPP_MIN_LOG_LEVEL=3
python3 app.py
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

<div align="center">
  FER2013 dataset by <a href="https://www.kaggle.com/msambare">msambare</a> on Kaggle
</div>
