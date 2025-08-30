# Hand Gesture Recognition (OpenCV + MediaPipe)

**Author:** Sneh Patel

A real-time hand gesture recognizer using MediaPipe Hands, OpenCV, and NumPy. It detects the four required gestures reliably and in real time:

- **Fist**
- **Thumbs Up**
- **Peace Sign**
- **Open Palm**

The system is orientation-robust (uses joint angles, not raw Y-coordinates) and includes temporal smoothing to avoid flickering predictions.

---

## Demonstration

<p align="center">
   <img src="src/demo.gif" alt="Demo" autoplay loop width="600" style="max-width:100%;height:auto;">
</p>

---

## Technology Justification

**Why MediaPipe Hands?**

- High-quality 21-landmark tracking per hand (wrist, knuckles, PIP, DIP, tips).
- Robust across lighting/orientations—greatly reduces custom preprocessing needs.
- Cross-platform (Windows, macOS, Linux) and easy Python integration.

**Why OpenCV?**

- Webcam capture & visualization: frame acquisition, color space conversion (BGR↔RGB), overlay text, and drawing utilities.
- Mature ecosystem and simple APIs for real-time apps.

**Why NumPy?**

- Vector math for fast, reliable angle computations (dot products, norms).
- Keeps the math concise and efficient.

**Alternatives considered:**

- Training a custom CNN/LSTM is heavier (data collection/labels, training time) and unnecessary since MediaPipe already provides excellent landmarks.
- Classical image heuristics (contours/skin color) are brittle across lighting/skin tones and camera quality.

---

## Gesture Logic Explanation

**Landmark & Angle-Based Method (Orientation-Robust):**

For each finger, we compute the angle at a key joint using three landmarks:

- **Thumb:** (2–3–4) → angle at landmark 3
- **Index:** (5–6–8) → angle at landmark 6
- **Middle:** (9–10–12) → angle at landmark 10
- **Ring:** (13–14–16) → angle at landmark 14
- **Pinky:** (17–18–20) → angle at landmark 18

Given points A–B–C, the angle at B is:

```
θ = arccos( (BA · BC) / (||BA|| * ||BC||) )
```

**Finger State Thresholds:**

- A finger is extended when its joint angle is ≥ ~160° (nearly straight).
- A finger is curled when its joint angle is < ~150° (noticeably bent).
- These thresholds are empirically chosen to be robust to natural hand variations and minor jitter.

**Disambiguation Rules (Order Matters):**
To avoid confusion between Fist and Thumbs Up, we evaluate gestures in a priority order:

1. **Thumbs Up:** Thumb extended, all other fingers curled → Return “Thumbs Up”
2. **Fist:** All four non-thumb fingers curled, thumb not extended → Return “Fist”
3. **Peace Sign:** Index & Middle extended, Ring & Pinky curled → Return “Peace Sign”
4. **Open Palm:** All five fingers extended → Return “Open Palm”
5. If no rule fires, return “Unknown Gesture.”

**Temporal Smoothing (Stability):**

- We keep a deque buffer of the last N detections (N=10 frames) and display the mode (most frequent). This suppresses flicker from transient misclassifications.
- Increase N for steadier output (slightly more latency).
- Decrease N for faster responsiveness (slightly less stable).

---

## Setup and Execution Instructions

### 1. Clone the repository (all platforms):

```
git clone <your-repo-link>
cd hand_gesture_recognition
```

### 2. Create a virtual environment:

**Windows:**

```
python -m venv .venv
```

**macOS/Linux:**

```
python3 -m venv .venv
```

### 3. Activate the virtual environment:

**Windows:**

```
.venv\Scripts\activate
```

**macOS/Linux:**

```
source .venv/bin/activate
```

### 4. Install dependencies (all platforms):

```
pip install -r requirements.txt
```

### 5. Run the application (all platforms):

```
python src/main.py
```

---

## Camera Tips

If you have multiple cameras and the wrong one opens, change:

```
cap = cv2.VideoCapture(0)
```

to `VideoCapture(1)` or `VideoCapture(2)`.

On macOS, you may need to grant camera permissions to your terminal/IDE under System Settings → Privacy & Security → Camera.

## Usage Notes & Tuning

**Thresholds:**

- Extended: ≥ 160°
- Curled: < 150°

You can fine-tune these in `recognize_gesture()` if your camera angle/hand size/environment is atypical.

**Smoothing Window:** `deque(maxlen=10)`

Increase for steadier labels; decrease for faster responsiveness.

**Performance:**

- Reduce camera resolution or process every 2nd frame if needed.
- Set `max_num_hands=1` if only one hand is required.

---

## Troubleshooting

**“No Hand” or frequent “Unknown Gesture”**

- Ensure good lighting and keep your hand fully in frame.
- Try reducing the distance to the camera.
- Slightly adjust thresholds (e.g., extended 155–165°).

**Lag or low FPS**

- Close other heavy apps.
- Lower capture resolution.
- Set `max_num_hands=1`.

---

## Brief Code References

- **Angle function:** vector math with dot product & arccos
- **Gesture rules:** priority order to separate Fist vs Thumbs Up
- **Deque smoothing:** mode over last N frames

---
