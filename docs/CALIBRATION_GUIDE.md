# Camera & Pitch Calibration Guide

## Purpose of Calibration

Single-camera tactical analysis requires transforming perspective coordinates from the broadcast video (where parallel lines appear to converge) into an orthographic, top-down 2D view of the pitch.

This perspective mapping is governed by a **Planar Homography Matrix** ($3 \times 3$), which requires a minimum of **4 point correspondences** between the video frame and known physical coordinates on the pitch.

```
Broadcast Video Frame (x, y)  ====== Homography (H) ======>  Tactical Minimap (x', y')
```

---

## Two-Window Calibration Tool (`calibrate.py`)

The repository includes an interactive dual-window calibration tool designed to make calibration straightforward:

```bash
python calibrate.py --video data/test_match.mp4 --output calibration.json
```

### How the Interface Works

When executed, two synchronized OpenCV windows will open:
1. **Video Frame**: Displays the first frame of your football match video.
2. **Pitch Diagram**: Displays a clean, standardized 2D FIFA football pitch ($105\text{m} \times 68\text{m}$) rendered to scale.

### Interactive Controls

| Action | Control | Description |
| :--- | :--- | :--- |
| **Select Point** | `Left Click` | Select a coordinate on the active window. |
| **Undo Last Selection** | `Right Click` | Remove the most recent point and revert to prior step. |
| **Save & Finish** | `Any Key` (after 4 pairs) | Saves points to `calibration.json` and closes windows. |
| **Cancel / Exit** | `Esc` / Close Window | Exits without saving. |

---

## Step-by-Step Calibration Walkthrough

1. **Launch the tool**:
   ```bash
   python calibrate.py
   ```
2. **Select Point 1 on the Video Frame**:
   - Locate an unambiguous pitch marking on the video frame (e.g., the intersection of the penalty box and the goal line).
   - `Left Click` precisely at that corner. A red marker with label `1` will appear.
3. **Select Point 1 on the Pitch Diagram**:
   - Look at the top status bar: it will prompt `"Now click the SAME point on the PITCH DIAGRAM (point 1)"`.
   - `Left Click` on the corresponding penalty box corner on the pitch diagram.
4. **Repeat for Points 2, 3, and 4**:
   - Alternate selecting on the video, then the pitch diagram.
5. **Save Calibration**:
   - Once all 4 pairs are selected, press any key on your keyboard.
   - `calibration.json` will be written to disk.

---

## Best Practices for Choosing Calibration Points

To maximize mathematical accuracy across the entire pitch and prevent distortion:

> [!TIP]
> **Maximize the Quadrilateral Area**:
> Select 4 points that span as wide an area of the pitch as possible. Points clustered close together will amplify projection errors outside their bounding polygon.

> [!IMPORTANT]
> **Avoid Collinear Points**:
> Never choose 3 points that lie along the same straight line (e.g., 3 points on the touchline). The homography calculation requires non-degenerate, 2D planar point distributions.

### Recommended Reference Landmark Combinations

1. **Penalty Box Marking Combination** (Ideal for footage focused on one half):
   - **Point 1**: Left intersection of 18-yard box and touchline / goal-line.
   - **Point 2**: Far corner of 18-yard box (pitch interior).
   - **Point 3**: Near corner of 18-yard box (pitch interior).
   - **Point 4**: Halfway line junction with touchline or penalty arc apex.

2. **Full-Pitch Combination** (Ideal for wide tactical camera feeds):
   - **Point 1**: Center spot $(52.5\text{m}, 34.0\text{m})$.
   - **Point 2**: Halfway line intersection with upper touchline $(52.5\text{m}, 0.0\text{m})$.
   - **Point 3**: Halfway line intersection with lower touchline $(52.5\text{m}, 68.0\text{m})$.
   - **Point 4**: Penalty spot of the defending or attacking side $(11.0\text{m}, 34.0\text{m})$.

---

## Pitch Coordinate Reference System

The pitch diagram uses the standard FIFA metric coordinate system:

| Landmark | Real-World Coordinates $(X, Y)$ in Meters |
| :--- | :--- |
| **Pitch Dimensions** | $105.0\text{m} \times 68.0\text{m}$ |
| **Top-Left Corner** | $(0.0, 0.0)$ |
| **Bottom-Right Corner** | $(105.0, 68.0)$ |
| **Center Spot** | $(52.5, 34.0)$ |
| **Center Circle Radius** | $9.15\text{m}$ |
| **Left Goal Line Center** | $(0.0, 34.0)$ |
| **Left Penalty Spot** | $(11.0, 34.0)$ |
| **Left 18-Yard Box** | $X \in [0, 16.5], Y \in [13.84, 54.16]$ |
| **Left 6-Yard Box** | $X \in [0, 5.5], Y \in [24.84, 43.16]$ |
| **Right Penalty Spot** | $(94.0, 34.0)$ |
| **Right 18-Yard Box** | $X \in [88.5, 105.0], Y \in [13.84, 54.16]$ |

---

## Inspecting `calibration.json`

The generated `calibration.json` structure:

```json
{
  "src_points": [
    [290, 560],
    [554, 432],
    [1744, 784],
    [12, 742]
  ],
  "dst_points_meters": [
    [16.94, 26.98],
    [16.79, 13.79],
    [42.87, 34.18],
    [17.99, 39.88]
  ],
  "pitch_diagram_scale": 6.67
}
```

- `src_points`: Pixel $(X, Y)$ coordinates in the source video.
- `dst_points_meters`: Physical coordinates in meters on the standardized pitch.
- `pitch_diagram_scale`: Scale factor (pixels/meter) used during calibration.

To re-run analysis with custom calibration:
```bash
python main.py --calibration calibration.json
```
