import { useState, useEffect, useRef } from "react";
import "./App.css";

const API_URL = "http://localhost:5000/api";

const emotionEmojis = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😊",
  sad: "😢",
  surprise: "😲",
  neutral: "😐",
};

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const detectionInterval = useRef(null);

  // Check backend health on mount
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((res) => res.json())
      .then((data) => console.log("Backend OK:", data))
      .catch(() =>
        setError("Backend not running on port 5000. Please start the server."),
      );

    return () => {
      stopDetection();
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsDetecting(true);
      setError(null);

      // Start capturing frames
      detectionInterval.current = setInterval(captureAndPredict, 2000);
    } catch (err) {
      setError("Camera error: " + err.message);
      setIsDetecting(false);
    }
  };

  const stopDetection = () => {
    setIsDetecting(false);

    // Stop camera stream
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    // Stop detection loop
    if (detectionInterval.current) {
      clearInterval(detectionInterval.current);
      detectionInterval.current = null;
    }

    setResult(null);
  };

  const captureAndPredict = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      // Draw current video frame to canvas
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const imageData = canvas.toDataURL("image/png");

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) throw new Error("API request failed");

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      // Don't show error to user for every failed frame to avoid spam
    }
  };

  // Helper to safely get emoji
  const getEmoji = (emotion) => {
    if (!emotion) return "😐";
    return emotionEmojis[emotion.toLowerCase()] || "😐";
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Facial Expression Recognition</h1>
        <p>Real-time emotion detection using machine learning</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="main-content">
        {/* Left Side: Camera */}
        <div className="camera-card">
          <div className="video-wrapper">
            {!isDetecting && (
              <div className="video-overlay">
                <svg
                  width="48"
                  height="48"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                  <circle cx="12" cy="13" r="4"></circle>
                </svg>
                <span>Camera Offline</span>
              </div>
            )}
            <video
              ref={videoRef}
              className="video-element"
              autoPlay
              playsInline
              muted
            />
            {/* Hidden canvas for off-screen capturing */}
            <canvas ref={canvasRef} style={{ display: "none" }} />
          </div>

          <div className="controls">
            {!isDetecting ? (
              <button className="btn btn-primary" onClick={startCamera}>
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                Start Detection
              </button>
            ) : (
              <button className="btn btn-secondary" onClick={stopDetection}>
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                </svg>
                Stop Detection
              </button>
            )}
          </div>
        </div>

        {/* Right Side: Results */}
        <div className="results-card">
          {!result ? (
            <div className="empty-state">
              <div className="empty-state-icon">✨</div>
              <h2>Ready to analyze</h2>
              <p>Start the camera to detect emotions in real-time</p>
            </div>
          ) : (
            <>
              <div className="primary-emotion">
                <div className="emoji-display animated">
                  {getEmoji(result.emotion)}
                </div>
                <h2 className="emotion-name">{result.emotion || "Unknown"}</h2>
                <div className="confidence-badge">
                  {result.confidence
                    ? `${result.confidence.toFixed(1)}% Confidence`
                    : "Analyzing..."}
                </div>
              </div>

              <div className="probabilities-list">
                {result.probabilities &&
                  Object.entries(result.probabilities)
                    .sort((a, b) => b[1] - a[1]) // Sort probabilities high to low
                    .map(([emotion, prob]) => (
                      <div key={emotion} className="prob-item">
                        <div className="prob-header">
                          <span>{emotion}</span>
                          <span>{prob.toFixed(1)}%</span>
                        </div>
                        <div className="prob-bar-container">
                          <div
                            className="prob-bar"
                            style={{
                              width: `${Math.max(0, Math.min(100, prob))}%`,
                            }}
                          ></div>
                        </div>
                      </div>
                    ))}
              </div>
            </>
          )}
        </div>
      </main>
      <footer className="footer">
        <p>
          &copy; <year>{new Date().getFullYear()}</year> Developed by Cokecaine. All rights reserved.
        </p>
      </footer>
    </div>
  );
}

export default App;
