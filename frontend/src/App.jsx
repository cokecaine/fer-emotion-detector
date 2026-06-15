import { useState, useEffect, useRef } from "react";
import "./App.css";

const API_URL = "http://localhost:5000/api";

const EMOTIONS = [
  { key: "happy",    emoji: "😊", color: "#e8a838", label: "Happy" },
  { key: "sad",      emoji: "😢", color: "#6b8cba", label: "Sad" },
  { key: "angry",    emoji: "😠", color: "#c05c3a", label: "Angry" },
  { key: "surprise", emoji: "😲", color: "#9b6ec8", label: "Surprise" },
  { key: "fear",     emoji: "😨", color: "#5aab8a", label: "Fear" },
  { key: "disgust",  emoji: "🤢", color: "#7a9e52", label: "Disgust" },
  { key: "neutral",  emoji: "😐", color: "#8a8a8a", label: "Neutral" },
];

const emotionMap = Object.fromEntries(EMOTIONS.map((e) => [e.key, e]));

/* ─── Icons ─────────────────────────────────────────────────────────────── */
const IconCamera = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
    <circle cx="12" cy="13" r="4"/>
  </svg>
);

const IconHome = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
    <polyline points="9 22 9 12 15 12 15 22"/>
  </svg>
);

const IconPlay = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <polygon points="5 3 19 12 5 21 5 3"/>
  </svg>
);

const IconStop = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <rect x="4" y="4" width="16" height="16" rx="3"/>
  </svg>
);

const IconArrow = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="5" y1="12" x2="19" y2="12"/>
    <polyline points="12 5 19 12 12 19"/>
  </svg>
);

const IconBrain = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-1.96-3 2.5 2.5 0 0 1-1.32-4.24 3 3 0 0 1 .34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.18-1.5z"/>
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 1.96-3 2.5 2.5 0 0 0 1.32-4.24 3 3 0 0 0-.34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.18-1.5z"/>
  </svg>
);

const IconZap = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
  </svg>
);

const IconShield = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  </svg>
);

/* ─── Navbar ─────────────────────────────────────────────────────────────── */
function Navbar({ page, setPage }) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handler);
    return () => window.removeEventListener("scroll", handler);
  }, []);

  return (
    <nav className={`navbar ${scrolled ? "scrolled" : ""}`}>
      <div className="navbar-inner">
        <button className="nav-logo" onClick={() => setPage("home")} id="nav-logo-btn">
          <span className="nav-logo-icon">◉</span>
          <span>FER - FaceRead</span>
        </button>

        <div className="nav-links">
          <button
            className={`nav-link ${page === "home" ? "active" : ""}`}
            onClick={() => setPage("home")}
            id="nav-home-btn"
          >
            <IconHome />
            Home
          </button>
          <button
            className={`nav-link ${page === "detect" ? "active" : ""}`}
            onClick={() => setPage("detect")}
            id="nav-detect-btn"
          >
            <IconCamera />
            Detect
          </button>
        </div>

        <button className="nav-cta" onClick={() => setPage("detect")} id="nav-cta-btn">
          Try it free
        </button>
      </div>
    </nav>
  );
}

/* ─── Home Page ──────────────────────────────────────────────────────────── */
function HomePage({ setPage }) {
  const FEATURES = [
    { icon: <IconBrain />, title: "Deep Learning Model", desc: "Trained on 35,000+ facial images using a custom CNN architecture for robust detection." },
    { icon: <IconZap />,   title: "Real-time Analysis",  desc: "Analyzes frames every 2 seconds directly from your webcam with low latency." },
    { icon: <IconShield />, title: "100% Private",       desc: "All processing happens locally. No data is stored or sent to external servers." },
  ];

  return (
    <div className="home-page">
      {/* Hero */}
      <section className="hero">
        <div className="hero-tag">
          <span className="hero-tag-dot" />
          AI-Powered Emotion Recognition
        </div>

        <h1 className="hero-title">
          Read the room
          <br />
          <span className="hero-title-accent">in real time.</span>
        </h1>

        <p className="hero-subtitle">
          FER - FaceRead uses deep learning to detect 7 distinct emotions from your webcam feed — instantly, privately, and with remarkable accuracy.
        </p>

        <div className="hero-actions">
          <button className="btn-hero-primary" onClick={() => setPage("detect")} id="hero-cta-btn">
            Start detecting
            <IconArrow />
          </button>
          <a className="btn-hero-secondary" href="https://github.com/cokecaine/fer-emotion-detector" target="_blank" rel="noopener noreferrer" id="hero-github-btn">
            View on GitHub
          </a>
        </div>

        {/* Floating emotion chips */}
        <div className="emotion-orbit" aria-hidden="true">
          {EMOTIONS.map((e, i) => (
            <div
              key={e.key}
              className="emotion-chip"
              style={{
                "--chip-color": e.color,
                "--delay": `${i * 0.55}s`,
                "--angle": `${(i / EMOTIONS.length) * 360}deg`,
              }}
            >
              <span className="chip-emoji">{e.emoji}</span>
              <span className="chip-label">{e.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Stats strip */}
      <section className="stats-strip">
        {[
          { value: "35K+",  label: "Training images" },
          { value: "7",     label: "Emotions detected" },
          { value: "~2s",   label: "Detection interval" },
          { value: "100%",  label: "On-device processing" },
        ].map((s) => (
          <div className="stat-item" key={s.label}>
            <span className="stat-value">{s.value}</span>
            <span className="stat-label">{s.label}</span>
          </div>
        ))}
      </section>

      {/* Emotion showcase */}
      <section className="showcase">
        <div className="section-tag">What we detect</div>
        <h2 className="section-title">Seven shades of feeling</h2>
        <p className="section-desc">From joy to surprise, our model captures the full spectrum of human expression.</p>

        <div className="emotion-grid">
          {EMOTIONS.map((e) => (
            <div className="emotion-card" key={e.key} style={{ "--card-color": e.color }}>
              <div className="emotion-card-emoji">{e.emoji}</div>
              <div className="emotion-card-name">{e.label}</div>
              <div className="emotion-card-bar" />
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="features">
        <div className="section-tag">Why FER - FaceRead</div>
        <h2 className="section-title">Built for precision.</h2>

        <div className="features-grid">
          {FEATURES.map((f) => (
            <div className="feature-card" key={f.title}>
              <div className="feature-icon">{f.icon}</div>
              <h3 className="feature-title">{f.title}</h3>
              <p className="feature-desc">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Banner */}
      <section className="cta-banner">
        <div className="cta-banner-inner">
          <h2>Ready to see it in action?</h2>
          <p>Open your camera and watch the model read your expressions in real time.</p>
          <button className="btn-hero-primary" onClick={() => setPage("detect")} id="banner-cta-btn">
            Open Detect
            <IconArrow />
          </button>
        </div>
      </section>
    </div>
  );
}

/* ─── Detect Page ────────────────────────────────────────────────────────── */
function DetectPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [backendOk, setBackendOk] = useState(null);
  const detectionInterval = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then(() => setBackendOk(true))
      .catch(() => {
        setBackendOk(false);
        setError("Backend not running on port 5000. Start the Flask server first.");
      });
    return () => stopDetection();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      if (videoRef.current) videoRef.current.srcObject = stream;
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
    if (videoRef.current?.srcObject) {
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
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const imageData = canvas.toDataURL("image/png");
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });
      if (!response.ok) throw new Error("API failed");
      setResult(await response.json());
    } catch (err) {
      console.error(err);
    }
  };

  const topEmotion = result?.emotion
    ? emotionMap[result.emotion.toLowerCase()]
    : null;

  const sortedProbs = result?.probabilities
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="detect-page">
      <div className="detect-header">
        <div className="detect-eyebrow">
          <span className={`status-dot ${backendOk === true ? "ok" : backendOk === false ? "err" : "idle"}`} />
          {backendOk === true ? "Backend connected" : backendOk === false ? "Backend offline" : "Connecting…"}
        </div>
        <h1 className="detect-title">Emotion Detector</h1>
        <p className="detect-sub">Point your camera at a face and let the model do the rest.</p>
      </div>

      {error && (
        <div className="error-banner" role="alert">
          <span>⚠</span>
          {error}
        </div>
      )}

      <div className="detect-grid">
        {/* Camera card */}
        <div className="d-card camera-card">
          <div className="d-card-header">
            <span className={`live-badge ${isDetecting ? "live" : ""}`}>
              <span className="live-dot" />
              {isDetecting ? "Live" : "Standby"}
            </span>
          </div>

          <div className={`video-wrap ${isDetecting ? "detecting" : ""}`}>
            {/* Corner brackets */}
            <span className="corner tl" />
            <span className="corner tr" />
            <span className="corner bl" />
            <span className="corner br" />
            {/* Scan line */}
            {isDetecting && <div className="scan-line" />}

            {!isDetecting && (
              <div className="video-placeholder">
                <div className="cam-icon-wrap">
                  <IconCamera />
                </div>
                <p>Camera offline</p>
                <span>Click Start Detection to begin</span>
              </div>
            )}

            <video
              ref={videoRef}
              className="video-el"
              autoPlay
              playsInline
              muted
            />
            <canvas ref={canvasRef} style={{ display: "none" }} />
          </div>

          <div className="cam-controls">
            {!isDetecting ? (
              <button className="btn-start" onClick={startCamera} id="btn-start-detection" disabled={backendOk === false}>
                <IconPlay />
                Start Detection
              </button>
            ) : (
              <button className="btn-stop-detection" onClick={stopDetection} id="btn-stop-detection">
                <IconStop />
                Stop
              </button>
            )}
          </div>
        </div>

        {/* Results card */}
        <div className="d-card results-card">
          <div className="d-card-header">
            <span className="d-card-label">Analysis</span>
          </div>

          {!result ? (
            <div className="empty-result">
              <div className="empty-orb">
                {EMOTIONS.slice(0, 4).map((e, i) => (
                  <span key={e.key} className="orb-emoji" style={{ "--oi": i }}>{e.emoji}</span>
                ))}
              </div>
              <h3>Awaiting input</h3>
              <p>Start the camera to detect emotions in real time</p>
            </div>
          ) : (
            <div className="result-content">
              {/* Primary emotion */}
              <div className="primary-result" style={{ "--emotion-color": topEmotion?.color || "#8a8a8a" }}>
                <div className="primary-emoji-wrap">
                  <span className="primary-emoji">{topEmotion?.emoji || "😐"}</span>
                </div>
                <div className="primary-text">
                  <h2 className="primary-label">{result.emotion || "Unknown"}</h2>
                  <div className="confidence-pill">
                    {result.confidence?.toFixed(1)}% confident
                  </div>
                </div>
              </div>

              {/* Probability bars */}
              <div className="probs-section">
                <p className="probs-heading">All emotions</p>
                <div className="probs-list">
                  {sortedProbs.map(([emotion, prob], idx) => {
                    const em = emotionMap[emotion.toLowerCase()];
                    return (
                      <div className={`prob-row ${idx === 0 ? "top" : ""}`} key={emotion}>
                        <div className="prob-meta">
                          <span className="prob-emoji">{em?.emoji || "❓"}</span>
                          <span className="prob-name">{emotion}</span>
                          <span className="prob-pct">{prob.toFixed(1)}%</span>
                        </div>
                        <div className="prob-track">
                          <div
                            className="prob-fill"
                            style={{
                              width: `${Math.max(0, Math.min(100, prob))}%`,
                              "--fill-color": em?.color || "#8a8a8a",
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─── App Root ───────────────────────────────────────────────────────────── */
export default function App() {
  const [page, setPage] = useState("home");

  // Smooth page transition
  const [transitioning, setTransitioning] = useState(false);
  const navigate = (target) => {
    if (target === page) return;
    setTransitioning(true);
    setTimeout(() => {
      setPage(target);
      setTransitioning(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }, 180);
  };

  return (
    <div className="app-root">
      <Navbar page={page} setPage={navigate} />
      <main className={`page-wrap ${transitioning ? "fading" : ""}`}>
        {page === "home" ? <HomePage setPage={navigate} /> : <DetectPage />}
      </main>
      <footer className="site-footer">
        <p>© {new Date().getFullYear()} FER - FaceRead — Facial Expression Recognition · Built with TensorFlow &amp; React</p>
      </footer>
    </div>
  );
}
