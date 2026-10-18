# Mathematical & Algorithmic Foundations

This document details the core computer vision and machine learning algorithms utilized across the **Tactical Football Match Analyzer**.

---

## 1. Planar Homography Transformation

### 1.1 Projective Geometry Formulation
A homography is an invertible mapping between two projective planes $\mathbb{P}^2 \to \mathbb{P}^2$. Given a point in homogeneous coordinates on the camera image plane $\mathbf{x} = [x, y, 1]^T$ and its corresponding point on the 2D pitch plane $\mathbf{x}' = [x', y', 1]^T$:

$$\mathbf{x}' \sim \mathbf{H}\mathbf{x} \implies \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Because $\mathbf{H}$ has 8 degrees of freedom (up to an arbitrary non-zero scale factor), 4 point correspondences provide 8 independent linear constraints:

$$\begin{aligned}
x' &= \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}} \\
y' &= \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}
\end{aligned}$$

### 1.2 Direct Linear Transformation (DLT)
For each correspondence $\mathbf{x}_i \leftrightarrow \mathbf{x}'_i$, the cross-product $\mathbf{x}'_i \times \mathbf{H}\mathbf{x}_i = \mathbf{0}$ yields:

$$\begin{bmatrix}
\mathbf{0}^T & -w'_i \mathbf{x}_i^T & y'_i \mathbf{x}_i^T \\
w'_i \mathbf{x}_i^T & \mathbf{0}^T & -x'_i \mathbf{x}_i^T
\end{bmatrix} \mathbf{h} = \mathbf{0}$$

Stacking constraints from 4 correspondences creates an $8 \times 9$ matrix $\mathbf{A}$. The solution $\mathbf{h}$ is the singular vector corresponding to the smallest singular value obtained via Singular Value Decomposition (SVD):

$$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T, \quad \mathbf{h} = \mathbf{v}_9$$

In this project, `cv2.findHomography` and `cv2.perspectiveTransform` evaluate and apply $\mathbf{H}$.

### 1.3 Foot Point Anchoring
A 3D object (a standing player) cannot be treated as a single point without choosing a reference plane. Because the homography maps the pitch surface ($Z = 0$), player coordinates must represent the point of contact with the ground:

$$\mathbf{p}_{\text{ground}} = \left(\frac{x_{\min} + x_{\max}}{2}, y_{\max}\right)$$

Selecting the top or center of the bounding box introduces severe perspective parallax, causing players far from the camera to be projected off the pitch.

---

## 2. Multi-Object Tracking with ByteTrack

The project tracks players and the ball across frames using ByteTrack.

### 2.1 The ByteTrack Association Strategy
Traditional trackers (e.g. SORT, DeepSORT) filter out bounding boxes below a high confidence threshold (e.g. 0.5), losing occluded or blurred players. ByteTrack retains **all** detection boxes:

1. **First Association**: Matches high-confidence detections ($D_{\text{high}}$) with active tracklets ($T$) using Kalman filter state prediction and bounding box Intersection over Union (IoU) distance.
2. **Second Association**: Takes remaining unmatched tracklets ($T_{\text{remain}}$) and matches them against low-confidence detections ($D_{\text{low}}$). This recovers occluded players whose confidence temporarily dipped.
3. **Track Lifecycle**: Tracklets without matches for a sustained window are marked lost and eventually deleted.

---

## 3. Team Assignment via Color Space Clustering

Separating two opposing teams and match officials requires robust color segmentation that resists environmental artifacts.

### 3.1 Region of Interest (ROI) Selection
Given a bounding box $[x_1, y_1, x_2, y_2]$, the jersey region is isolated:

$$\text{ROI} = \text{Frame}\left[y_1 : y_1 + \left\lfloor \frac{y_2 - y_1}{2} \right\rfloor, x_1 : x_2\right]$$

This avoids confounding pixels from:
- Team shorts and socks.
- Field turf.
- Contact shadows cast on the grass.

### 3.2 Intra-Player K-Means
$K\text{-Means}$ with $K=3$ partitions the ROI pixel array into 3 color centroids $\mathbf{c}_1, \mathbf{c}_2, \mathbf{c}_3$ in BGR color space. To select the true jersey cluster rather than skin tone or residual grass:

Each centroid is mapped to HSV space $(h, s, v)$:
$$\text{Penalty}_{\text{grass}} = \begin{cases} 200 & \text{if } 30 \le h \le 90 \land s > 40 \land v > 40 \\ 0 & \text{otherwise} \end{cases}$$
$$\text{Penalty}_{\text{shadow}} = \begin{cases} 150 & \text{if } v < 50 \\ 0 & \text{otherwise} \end{cases}$$
$$\text{Score}(\mathbf{c}_i) = s_i + 0.5 \cdot v_i - \text{Penalty}_{\text{grass}} - \text{Penalty}_{\text{shadow}}$$

The centroid maximizing this score is returned as the player's representative jersey color $\mathbf{C}_p$.

### 3.3 Inter-Player Classification (Global Clustering)
Across all observed players, an aggregate matrix $\mathbf{X} \in \mathbb{R}^{N \times 3}$ is formed. Secondary $K\text{-Means}(K=3)$ identifies:
- **Referee Cluster**: Centroid with the lowest saturation-brightness product:
  $$\text{argmin}_k \left(S_k \times V_k\right)$$
- **Team A & Team B Clusters**: The remaining two clusters, mapped to high-contrast tactical pigments (e.g. Red and Yellow).

---

## 4. Ball Tracking & Velocity-Based Extrapolation

Because the sports ball is small ($22\text{cm}$ diameter) and moves at high velocity (up to $30\text{m/s}$), YOLO detections intermittently miss the ball during rapid motion blur.

### 4.1 Velocity Smoothing
When a ball detection $\mathbf{b}_t = [x_1, y_1, x_2, y_2]$ is received:
$$\mathbf{c}_t = \left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$$
$$\mathbf{v}_{\text{inst}} = \mathbf{c}_t - \mathbf{c}_{t-1}$$
$$\mathbf{v}_t = \alpha \mathbf{v}_{t-1} + (1 - \alpha) \mathbf{v}_{\text{inst}}, \quad \alpha = 0.6$$

### 4.2 Damped Extrapolation
When YOLO fails to detect the ball for $k$ frames ($k \le k_{\max}$):
$$\mathbf{v}_{t+k} = \mathbf{v}_{t+k-1} \cdot \gamma, \quad \gamma = 0.92$$
$$\mathbf{b}_{t+k} = \mathbf{b}_{t+k-1} + [\Delta x, \Delta y, \Delta x, \Delta y]$$

This maintains visual continuity and tactical minimap trajectory without freezing the ball in place during a fast pass or cross.
