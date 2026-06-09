import { useState, useEffect, useRef } from "react";
import "./App.css";

const API_URL = "http://localhost:5000/api";

const emotionEmojis = {
  angry:    "😠",
  disgust:  "🤢",
  fear:     "😨",
  happy:    "😊",
  sad:      "😢",
  surprise: "😲",
  neutral:  "😐",
};

// Camera icon (inline SVG)
const CameraIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
       stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
    <circle cx="12" cy="13" r="4"/>
  </svg>
);

// Play icon
const PlayIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <polygon points="5 3 19 12 5 21 5 3"/>
  </svg>
);

// Stop icon
const StopIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <rect x="4" y="4" width="16" height="16" rx="2"/>
  </svg>
);

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
        setError("Backend not running on port 5000. Please start the server.")
      );

    return () => {
      stopDetection();
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsDetecting(true);
      setError(null);
      detectionInterval.current = setInterval(captureAndPredict, 2000);
    } catch (err) {
      setError("Camera error: " + err.message);
      setIsDetecting(false);
    }
  };

  const stopDetection = () => {
    setIsDetecting(false);
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
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
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const imageData = canvas.toDataURL("image/png");
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });
      if (!response.ok) throw new Error("API request failed");
      setResult(await response.json());
    } catch (err) {
      console.error(err);
    }
  };

  const getEmoji = (emotion) => {
    if (!emotion) return "😐";
    return emotionEmojis[emotion.toLowerCase()] || "😐";
  };

  return (
    <div className="app-container">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-eyebrow">
          <span className="dot" />
          AI Emotion Analysis
        </div>
        <h1>Facial Expression Recognition</h1>
        <p>Real-time emotion detection powered by deep learning</p>
      </header>

      {/* ── Error ── */}
      {error && <div className="error-banner">{error}</div>}

      {/* ── Main ── */}
      <main className="main-content">

        {/* Left – Camera */}
        <div className="camera-card">
          <div className="card-label">
            <span className={`badge ${isDetecting ? "live" : ""}`} />
            {isDetecting ? "Live feed" : "Camera"}
          </div>

          <div className={`video-wrapper ${isDetecting ? "is-detecting" : ""}`}>
            {/* Scanning line overlay when active */}
            <div className="video-scan-line" />

            {/* All-four-corners decoration */}
            <div className="video-corners" />

            {!isDetecting && (
              <div className="video-overlay">
                <div className="video-overlay-icon">
                  <CameraIcon />
                </div>
                <span className="video-overlay-text">Camera offline</span>
                <span className="video-overlay-sub">Click start to begin</span>
              </div>
            )}

            <video
              ref={videoRef}
              className="video-element"
              autoPlay
              playsInline
              muted
            />
            <canvas ref={canvasRef} style={{ display: "none" }} />
          </div>

          <div className="controls">
            {!isDetecting ? (
              <button className="btn btn-primary" onClick={startCamera} id="btn-start">
                <PlayIcon />
                Start Detection
              </button>
            ) : (
              <button className="btn btn-stop" onClick={stopDetection} id="btn-stop">
                <StopIcon />
                Stop Detection
              </button>
            )}
          </div>
        </div>

        {/* Right – Results */}
        <div className="results-card">
          <div className="card-label">
            <span className="badge" />
            Analysis
          </div>

          {!result ? (
            <div className="empty-state">
              <div className="empty-state-orb">✨</div>
              <h2>Ready to analyze</h2>
              <p>Start the camera to detect emotions in real-time</p>
            </div>
          ) : (
            <>
              {/* Primary emotion display */}
              <div className="primary-emotion">
                <div className="emoji-ring">
                  <span className="emoji-display">{getEmoji(result.emotion)}</span>
                </div>
                <h2 className="emotion-name">{result.emotion || "Unknown"}</h2>
                <div className="confidence-badge">
                  {result.confidence
                    ? `${result.confidence.toFixed(1)}% confidence`
                    : "Analyzing…"}
                </div>
              </div>

              {/* Probability bars */}
              {result.probabilities && (
                <>
                  <p className="prob-section-label">All emotions</p>
                  <div className="probabilities-list">
                    {Object.entries(result.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([emotion, prob], idx) => (
                        <div
                          key={emotion}
                          className={`prob-item ${idx === 0 ? "is-top" : ""}`}
                        >
                          <div className="prob-header">
                            <span>
                              <span className="emotion-emoji">
                                {getEmoji(emotion)}
                              </span>
                              {emotion}
                            </span>
                            <span className="prob-pct">{prob.toFixed(1)}%</span>
                          </div>
                          <div className="prob-bar-container">
                            <div
                              className="prob-bar"
                              style={{ width: `${Math.max(0, Math.min(100, prob))}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        <p>© {new Date().getFullYear()} Cokecaine — Facial Expression Recognition</p>
      </footer>
    </div>
  );
}

export default App;
