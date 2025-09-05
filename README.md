
# Sign Language Translator (MediaPipe + RandomForest + Flask)

A real-time, offline-capable ASL translator for bidirectional communication:
- **Recognition**: Webcam → MediaPipe hand landmarks → RandomForest classifier → text (with sentence buffer).
- **Text-to-Sign**: Typed text → playlist of sign videos (word MP4s with letter fallback).

## Features
- Low-latency CPU inference (no GPU required)
- Offline operation (local models & videos)
- Flask web UI (`/video` for live, `/play` for text-to-sign)
- Adjustable confidence threshold (default: 0.8) & temporal smoothing

## Project Structure
```
.
├── app.py                     # Flask app (routes: /, /video, /play)
├── datacollection.py          # Capture labeled samples
├── createdataset.py           # Convert frames → landmarks (data.pickle)
├── datatraining.py            # Train RandomForest → models/model.p
├── datatesting.py             # Live recognition (overlay, buffer)
├── generatesign.py            # Text → video playlist mapping
├── config.json                # Paths, thresholds, smoothing, camera index
├── models/
│   └── model.p                # Trained RandomForest model (LFS)
├── words/                     # MP4 clips for words & letters (LFS)
│   └── README.md
├── data/                      # (Optional) raw/processed data (gitignored)
├── static/ templates/         # Flask assets (if used)
├── requirements.txt
├── .gitattributes             # Git LFS tracking
├── .gitignore
└── LICENSE
```

## Quick Start
```bash
# 1) Create & activate venv
python3 -m venv .venv && source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate                             # Windows PowerShell

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Train model
python datacollection.py
python createdataset.py
python datatraining.py

# 4) Run app
python app.py
# Open http://localhost:5000
```

## Configuration
Edit `config.json`:
```json
{
  "camera_index": 0,
  "threshold": 0.8,
  "smoothing_window": 5,
  "words_dir": "words",
  "model_path": "models/model.p"
}
```

## Large Files (Videos/Models)
This repo uses **Git LFS** to track `*.mp4`, `*.mov`, `*.mkv`, `*.avi`, `*.p`, and `*.pickle`. If you prefer not to store videos in git, publish them as a **GitHub Release** and keep only `words/README.md` with download instructions.

## License
MIT — see `LICENSE`.

## Acknowledgements
- MediaPipe Hands for landmark extraction
- scikit-learn RandomForestClassifier
- OpenCV for video I/O
- Flask for the web interface
