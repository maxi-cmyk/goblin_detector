# 🧌 Spine Warden: Goblin Mode Detector

A real-time posture detection app that uses your webcam and AI to keep you sitting straight. Slouch too long and your HP drains — hit zero and you get hit with the **shrimp posture meme** and a **goblin face overlay** as punishment.

## How It Works

1. **YOLOv8 Pose Estimation** detects your nose and shoulder keypoints in real-time
2. A **posture ratio** (nose height vs. shoulder width) is calculated each frame
3. Press **C** to calibrate your "good posture" baseline
4. Slouch below the baseline → HP drains → hit 0 HP → **SHRIMP'D** 🦐
5. After the 5-second shrimp flash, a **goblin face** (horns, ears, fangs) is drawn over you for 5 more seconds of shame

## Features

- ⚡ Real-time pose detection via YOLOv8
- 🎮 HP bar with smooth green → yellow → red gradient
- 🦐 Fullscreen shrimp meme on game over
- 🔊 Cat laugh sound effect
- 🧌 Goblin face overlay as post-game-over shame
- 🔥 Good posture streak counter + best streak tracking
- 📊 Session stats summary on quit (duration, posture %, best streak, game overs)
- ☕ Break reminder every 30 minutes
- 🔄 Re-calibrate anytime by pressing **C**

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download YOLO pose model (auto-downloads on first run)
python main.py
```

## Controls

| Key | Action                     |
| --- | -------------------------- |
| `C` | Calibrate posture baseline |
| `Q` | Quit                       |

## Project Structure

```
goblin_detector/
├── main.py              # Main loop (camera, inference, rendering)
├── core/
│   ├── detector.py      # YOLOv8 pose keypoint extraction
│   ├── geometry.py      # Posture ratio calculation
│   └── state.py         # HP, calibration, slouch detection logic
├── ui/
│   └── renderer.py      # HP bar, shrimp flash, goblin face overlay
├── assets/
│   ├── shrimp_posture.png
│   └── cat-laugh-meme-1.mp3
└── requirements.txt
```

## Requirements

- Python 3.10+
- Webcam
- macOS / Linux / Windows
