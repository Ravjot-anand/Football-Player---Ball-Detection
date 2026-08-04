# AI Football Player & Ball Detection and Tracker

> **My First YOLO Computer Vision Project**  
> An end-to-end computer vision pipeline for football match video analysis — detecting players, referees, and the ball, tracking player movements across frames, clustering team jersey colors, and interpolating ball trajectories.

---

## Demo & Overview

<video src="https://github.com/user-attachments/assets/7f9014d0-e271-4ea0-b20f-de23ab183d67" controls="controls" width="100%"></video>

---

## Project Highlights

- **Custom Object Detection**: Fine-tuned **YOLO11** model trained to detect 4 classes: `player`, `goalkeeper`, `referee`, and `ball`.
- **Multi-Object Tracking (MOT)**: Integrated **Supervision ByteTrack** for persistent player ID tracking across video frames.
- **Robust Team Color Clustering**: 
  - Upper-torso chest cropping with **HSV color space masking** to filter out green pitch grass.
  - Multi-frame sampling and **K-Means clustering** ($k=2$) with majority voting per player track to prevent team flickering across frames.
  - Goalkeeper jersey override support for distinct kit colors.
- **Ball Trajectory Interpolation**: Uses Pandas quadratic/linear interpolation to smooth and reconstruct missing ball detections across fast-moving frames.
- **Ball Possession Attribution**: Calculates minimum Euclidean distance between player foot coordinates and the ball position to attribute possession dynamically.

---

## Understanding YOLO (You Only Look Once)

### How YOLO Works
Unlike traditional multi-stage detectors (like R-CNN) that first generate region proposals and then classify them, **YOLO (You Only Look Once)** treats object detection as a single regression problem:

1. **Single-Pass Architecture**: YOLO passes the input image through a neural network once, predicting bounding boxes and class probabilities simultaneously across the entire image grid.
2. **Grid-Based Prediction**: The image is divided into an $S \times S$ grid. Each cell is responsible for predicting bounding boxes and confidence scores for objects whose centers fall inside that grid cell.
3. **Anchor Boxes & Multi-Scale Detection**: YOLO utilizes feature pyramids and anchor boxes to detect objects of varying scales and aspect ratios (from small footballs to full-height players).
4. **Real-Time Efficiency**: By framing detection as a single pass, YOLO achieves ultra-fast inference speeds, making it ideal for video analysis.

### YOLO Model Used
This project utilizes **YOLO11** (fine-tuned from `yolo11x.pt`), trained on a custom football detection dataset.

---

## Project Architecture & Pipeline

```
                              ┌────────────────────────┐
                              │    Input Football      │
                              │      Video File        │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Frame Extraction (cv2) │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │  YOLO11 Detection      │
                              │ (Player, Referee, Ball)│
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Supervision ByteTrack  │
                              │ (Persistent Track IDs) │
                              └───────────┬────────────┘
                                          │
                ┌─────────────────────────┼─────────────────────────┐
                │                         │                         │
                ▼                         ▼                         ▼
   ┌────────────────────────┐┌────────────────────────┐┌────────────────────────┐
   │ HSV Grass Masking &    ││  Pandas Interpolation  ││ Distance-Based Player  │
   │ Multi-Frame KMeans Team││ for Missing Ball BBoxes││   Ball Possession      │
   │    Color Assignment    │└───────────┬────────────┘└───────────┬────────────┘
   └────────────┬───────────┘            │                         │
                │                         └────────────┬────────────┘
                └─────────────────────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Dynamic Frame Renderer │
                              │ (Ellipses, IDs, Tags)  │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │     Output Video       │
                              │ (output-videos/*.avi)  │
                              └────────────────────────┘
```

---

## Repository Structure

```
football/
├── input-videos/              <-- Add input videos here (Ignored by Git)
│   └── .gitkeep
├── output-videos/             <-- Generated output videos saved here (Ignored by Git)
│   └── .gitkeep
├── models/                    <-- Place trained YOLO weights here (.pt weights ignored by Git)
│   └── .gitkeep
├── stubs/                     <-- Cached detection pickle stubs
│   └── .gitkeep
├── Player_Ball_Assigner/      <-- Ball possession logic module
│   └── player_ball_assigner.py
├── Teams/                     <-- Team color extraction & KMeans clustering module
│   └── team_assigner.py
├── trackers/                  <-- YOLO & ByteTrack tracking module
│   └── tracker.py
├── utils/                     <-- Video I/O & Bounding Box geometry helper utilities
│   ├── bbox_utils.py
│   └── video_utils.py
├── main.py                    <-- Main execution script
├── yolo_inference.py          <-- Quick inference test script
├── training_yolo_v11.ipynb    <-- Jupyter notebook for downloading dataset & training YOLO11
├── .gitignore                 <-- Standard gitignore for weights, videos & caches
└── README.md
```

> **Note**: Large model checkpoints (`*.pt`), video files (`*.mp4`, `*.avi`), and pickle stubs (`*.pkl`) are excluded from GitHub via `.gitignore` to maintain a lightweight repository.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/football-tracker.git
cd football-tracker
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core packages directly:
pip install ultralytics supervision opencv-python numpy pandas scikit-learn roboflow
```

---

## Dataset Download & Model Training

### 1. Download Dataset from Roboflow
The dataset contains labeled images for players, referees, goalkeepers, and footballs. 

You can download it programmatically using the Roboflow API as shown in `training_yolo_v11.ipynb`:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov11")
```

### 2. Train YOLO Model
Open `training_yolo_v11.ipynb` or run the following training script using Ultralytics:

```python
from ultralytics import YOLO

# Load base YOLO11 pretrained model
model = YOLO("yolo11x.pt")

# Train model on downloaded football dataset
results = model.train(
    data="football-players-detection-1/data.yaml",
    epochs=10,
    imgsz=640,
    batch=4
)
```

After training completes, copy the generated best weights file to the `models/` directory:
```bash
cp runs/detect/train/weights/best.pt models/best.pt
```

---

## Running the Tracker

### 1. Add Input Video
Place your raw match video file in the `input-videos/` directory:
```bash
cp /path/to/your/match_video.mp4 input-videos/08fd33_4.mp4
```

### 2. Run the Main Script
Execute `main.py` to run detection, tracking, team clustering, ball interpolation, and annotation rendering:
```bash
python main.py
```

### 3. View Results
The processed output video with annotated player ellipses, track IDs, team colors, and ball possession markers will be generated inside `output-videos/`:
```bash
output-videos/08fd33_4_output.avi
```

---

## Tech Stack

- **Computer Vision**: OpenCV, Ultralytics YOLO11, Supervision (ByteTrack)
- **Machine Learning**: Scikit-Learn (K-Means Clustering)
- **Data & Math**: NumPy, Pandas
- **Language**: Python 3.10+

---

## Acknowledgments & Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for state-of-the-art real-time object detection models.
- [Roboflow](https://roboflow.com/) for dataset hosting and annotations.
- [Supervision](https://github.com/roboflow/supervision) for ByteTrack tracking implementation.
