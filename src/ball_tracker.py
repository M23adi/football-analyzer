import numpy as np

class BallTracker:
    """
    Temporal memory and trajectory estimator for the football.
    If YOLO loses the ball due to motion blur or occlusion, this module maintains
    ball continuity and predicts future positions using damped velocity extrapolation.
    """
    def __init__(self, max_missing_frames=12, damping=0.92):
        self.last_position = None
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.missing_frames = 0
        self.max_missing_frames = max_missing_frames
        self.damping = damping
        self.trajectory_history = []  # List of center (x, y) coordinates
        self.max_history = 30

    def update(self, detected_box):
        """
        Updates the ball's position. Pass 'None' if YOLO didn't detect the ball in current frame.
        
        Args:
            detected_box: Array-like [x1, y1, x2, y2] or None
            
        Returns:
            np.ndarray or None: Filtered [x1, y1, x2, y2] bounding box
        """
        if detected_box is not None:
            curr_box = np.array(detected_box, dtype=np.float32)
            curr_center = np.array([
                (curr_box[0] + curr_box[2]) / 2.0,
                (curr_box[1] + curr_box[3]) / 2.0
            ], dtype=np.float32)
            
            if self.last_position is not None and self.missing_frames == 0:
                prev_center = np.array([
                    (self.last_position[0] + self.last_position[2]) / 2.0,
                    (self.last_position[1] + self.last_position[3]) / 2.0
                ], dtype=np.float32)
                # Instantaneous velocity
                instant_v = curr_center - prev_center
                # Exponential moving average for velocity to avoid noise spikes
                self.velocity = 0.6 * self.velocity + 0.4 * instant_v
            
            self.last_position = curr_box
            self.missing_frames = 0
            self._record_history(curr_center)
            return self.last_position

        # Ball is missed
        self.missing_frames += 1
        if self.missing_frames <= self.max_missing_frames and self.last_position is not None:
            # Extrapolate position using damped velocity
            self.velocity *= self.damping
            dx, dy = self.velocity[0], self.velocity[1]
            extrapolated_box = self.last_position.copy()
            extrapolated_box[0] += dx
            extrapolated_box[1] += dy
            extrapolated_box[2] += dx
            extrapolated_box[3] += dy
            self.last_position = extrapolated_box
            
            center = np.array([
                (extrapolated_box[0] + extrapolated_box[2]) / 2.0,
                (extrapolated_box[1] + extrapolated_box[3]) / 2.0
            ], dtype=np.float32)
            self._record_history(center)
            return self.last_position

        # Lost for too long: reset state
        self.last_position = None
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        return None

    def _record_history(self, center):
        self.trajectory_history.append((int(center[0]), int(center[1])))
        if len(self.trajectory_history) > self.max_history:
            self.trajectory_history.pop(0)

    def get_trajectory(self):
        """Returns the recent center coordinates of the ball trajectory."""
        return list(self.trajectory_history)