# Project Memory: Tactical Football Match Analyzer

## 1. Project Identity & Purpose
- **Repository**: `football-analyzer`
- **Core Mission**: Build an end-to-end computer vision tactical analysis system that converts broadcast football (soccer) video footage into real-time 2D pitch coordinates, clusters players by jersey color into teams and officials, tracks the ball through occlusions, and renders a synchronized tactical minimap radar.
- **Target Hardware**: Windows 11, NVIDIA RTX 3050 (CUDA 12.1), Intel/AMD x64 CPU. Supports automatic CPU fallback for portable execution.
- **Python Environment**: Python 3.11 / 3.12, PyTorch 2.5+, Ultralytics 8.4+, OpenCV 4.13+, Scikit-Learn 1.9+.

---

## 2. Codebase Map & Module Responsibilities

```
football-analyzer/
├── main.py                      # Main pipeline orchestrator, CLI parser, video I/O, live HUD
├── calibrate.py                 # Interactive two-window camera homography calibration tool
├── calibration.json             # Stored 4-point video-to-pitch calibration mapping
├── requirements.txt             # Project dependencies (pinned with compatible versions)
├── .gitignore                   # Git exclusion rules (prevents tracking venv, weights, cache)
├── MEMORY.md                    # This persistent project memory and architectural journal
├── README.md                    # Primary repository overview, visual demo, and user guide
│
├── src/
│   ├── __init__.py              # Package initializer
│   ├── detection.py             # MatchAnalyzer: YOLOv8 + ByteTrack multi-object tracker
│   ├── ball_tracker.py          # BallTracker: Temporal memory, momentum velocity, trajectory buffer
│   ├── team_assigner.py         # TeamAssigner: K-Means ROI jersey color extraction & clustering
│   ├── camera_transformation.py # CameraTransformation: Planar homography & inverse projection
│   └── minimap.py               # Minimap: Scaled 2D FIFA football pitch renderer with trails
│
├── data/
│   └── test_match.mp4           # Benchmark broadcast video clip (Brazil vs Switzerland)
│
├── assets/
│   └── demo_preview.jpg         # Visual screenshot showing side-by-side tactical minimap
│
└── docs/
    ├── ARCHITECTURE.md          # In-depth system architecture and dataflow diagrams
    ├── CALIBRATION_GUIDE.md     # Comprehensive guide for two-window pitch calibration
    └── ALGORITHMS.md            # Mathematical derivations of homography, ByteTrack, & K-Means
```

---

## 3. Key Design Decisions & Architectural Rationale

### A. Planar Homography ($3 \times 3$) vs. Full 3D Camera Pose (PnP)
- **Decision**: Used planar homography via 4 point correspondences (`cv2.findHomography`).
- **Rationale**: A football pitch is mathematically planar ($Z = 0$). Solving full 6-DoF camera extrinsics ($R, t$) and intrinsics ($K$) requires either camera focal length calibration or vanishing line detection, which is brittle on broadcast footage with varying optical zoom. Homography provides exact metric transformation directly to the 2D pitch without requiring internal camera parameters.

### B. Foot-Point Ground Contact Mapping
- **Decision**: Projected $(\frac{x_1 + x_2}{2}, y_2)$ instead of bounding box centers.
- **Rationale**: In perspective projection, objects with physical height (like players standing $1.8\text{m}$ tall) project upward away from the ground plane. Only the point where their cleats contact the turf lies on the ground plane ($Z = 0$).

### C. Top-Half Bounding Box Cropping for Jersey Color
- **Decision**: Crop top 50% of the player bounding box and penalize green/shadow pixels.
- **Rationale**: Lower halves contain shorts, socks, and significant amounts of green turf pixels. Running K-Means on the full box frequently picked up green grass as the dominant color.

### D. Velocity-Damped Ball Extrapolation
- **Decision**: Added an Exponential Moving Average (EMA) velocity vector with damping ($0.92$) when YOLO loses the ball.
- **Rationale**: Fast passes or aerial crosses suffer from motion blur where YOLO confidence drops below threshold for 2–6 frames. Freezing the ball in place produced jarring stutter; linear extrapolation maintains smooth physical motion.

### E. Automatic Device Selection
- **Decision**: Check `torch.cuda.is_available()` dynamically in `src/detection.py` and accept an explicit `--device` CLI flag.
- **Rationale**: Hardcoded `.to('cuda')` caused immediate crashes when running on CPU-only machines, cloud runners, or development environments without an active CUDA runtime.

---

## 4. Completed Modules vs. Future Roadmap

### Completed & Validated
- [x] YOLOv8 + ByteTrack persistent multi-object tracking for players and ball.
- [x] Interactive two-window homography calibration tool (`calibrate.py`).
- [x] Planar homography projection (`src/camera_transformation.py`) with batch processing and inverse mapping.
- [x] K-Means jersey color extraction with green and dark shadow penalty (`src/team_assigner.py`).
- [x] Global player clustering into Team A, Team B, and Referee clusters.
- [x] Ball memory tracking with velocity smoothing and trajectory buffering (`src/ball_tracker.py`).
- [x] Standardized FIFA 2D pitch minimap ($105\text{m} \times 68\text{m}$) with grass stripes and ball trails (`src/minimap.py`).
- [x] CLI arguments with `argparse`, video export (`--output`), and headless mode (`--no-display`) in `main.py`.
- [x] Tested and verified on NVIDIA RTX 3050 GPU (11.6+ FPS sustained).

### Suggested Future Roadmap / Next Milestones
- [ ] **Dynamic Camera Motion Compensation**: Broadcast cameras pan, tilt, and zoom during active play. Integrating SIFT/ORB feature matching or Optical Flow across successive background frames will allow dynamic updating of the homography matrix $\mathbf{H}_t$.
- [ ] **Voronoi Pitch Control**: Compute Voronoi diagrams on the 2D pitch using player coordinates to visualize tactical space domination and passing lanes.
- [ ] **Player Physical Metrics**: Convert meter coordinates over consecutive frames into real-time sprint speeds ($\text{km/h}$), total distance covered, and high-intensity sprint counts.
- [ ] **Player Tactical Heatmaps**: Accumulate Gaussian density distributions over the 2D pitch to generate individual and team position heatmaps.

---

## 5. Environment & Quickstart Reference

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run interactive calibration (if calibrating a new video)
python calibrate.py --video data/test_match.mp4 --output calibration.json

# Run live tactical analysis
python main.py

# Export analyzed video without GUI
python main.py --output output_tactical.mp4 --no-display
```
