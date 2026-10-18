import cv2
import numpy as np

class Minimap:
    """
    Renders an accurate 2D top-down tactical representation of a standard football pitch (105m x 68m).
    Projects player feet positions, team jersey colors, player track IDs, and ball trajectory.
    """
    def __init__(self, width=682, height=442):
        self.width = width
        self.height = height
        
        # Scale factor (pixels per meter) — derived from standard pitch dimensions
        self.scale = width / 105.0
        
        # Base pitch (dark green with alternating grass stripes)
        self.base_pitch = np.full((self.height, self.width, 3), (34, 120, 34), dtype=np.uint8)
        
        # Add alternating vertical grass stripes for broadcast realism
        stripe_w = max(1, self.width // 10)
        for i in range(0, self.width, stripe_w * 2):
            self.base_pitch[:, i:min(self.width, i + stripe_w)] = (34, 139, 34)
        
        # Draw pitch lines and markings
        self._draw_lines()

    def _draw_lines(self):
        """Draw a full football pitch with standard FIFA markings (landscape orientation)."""
        c = (255, 255, 255)   # White line color
        t = 2                  # Line thickness
        s = self.scale
        
        # Helper to convert (meters_x, meters_y) to pixel coords
        def m(x, y):
            return (int(round(x * s)), int(round(y * s)))
        
        # Pitch outline
        cv2.rectangle(self.base_pitch, m(0, 0), m(105, 68), c, t)
        
        # Halfway line (vertical)
        cv2.line(self.base_pitch, m(52.5, 0), m(52.5, 68), c, t)
        
        # Center circle (radius 9.15m) and center spot
        cv2.circle(self.base_pitch, m(52.5, 34), int(round(9.15 * s)), c, t, lineType=cv2.LINE_AA)
        cv2.circle(self.base_pitch, m(52.5, 34), 4, c, -1, lineType=cv2.LINE_AA)
        
        # --- Left Side (Home / Penalty Box) ---
        pen_top = (68 - 40.32) / 2  # 13.84m
        pen_bot = pen_top + 40.32    # 54.16m
        cv2.rectangle(self.base_pitch, m(0, pen_top), m(16.5, pen_bot), c, t)
        
        goal_top = (68 - 18.32) / 2
        goal_bot = goal_top + 18.32
        cv2.rectangle(self.base_pitch, m(0, goal_top), m(5.5, goal_bot), c, t)
        
        cv2.circle(self.base_pitch, m(11, 34), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(self.base_pitch, m(11, 34), (int(round(9.15 * s)), int(round(9.15 * s))),
                     0, -53, 53, c, t, lineType=cv2.LINE_AA)
        
        # --- Right Side (Away / Penalty Box) ---
        cv2.rectangle(self.base_pitch, m(105 - 16.5, pen_top), m(105, pen_bot), c, t)
        cv2.rectangle(self.base_pitch, m(105 - 5.5, goal_top), m(105, goal_bot), c, t)
        cv2.circle(self.base_pitch, m(94, 34), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(self.base_pitch, m(94, 34), (int(round(9.15 * s)), int(round(9.15 * s))),
                     0, 127, 233, c, t, lineType=cv2.LINE_AA)

    def draw_points(self, player_points, ball_point, player_colors, ball_history_2d=None):
        """
        Draw players and ball on the minimap with optional ball trajectory trail.
        
        Args:
            player_points (dict): {track_id: (x, y)} transformed coordinates.
            ball_point (tuple or None): (x, y) transformed coordinate of the ball.
            player_colors (dict): {track_id: (B, G, R)}
            ball_history_2d (list, optional): List of transformed 2D ball (x, y) coordinates for trail.
            
        Returns:
            np.ndarray: The drawn minimap frame.
        """
        pitch = self.base_pitch.copy()
        
        # Draw ball trajectory trail if available
        if ball_history_2d and len(ball_history_2d) > 1:
            for i in range(1, len(ball_history_2d)):
                pt1 = ball_history_2d[i - 1]
                pt2 = ball_history_2d[i]
                if pt1 is not None and pt2 is not None:
                    alpha = i / len(ball_history_2d)
                    thickness = max(1, int(round(3 * alpha)))
                    cv2.line(pitch, pt1, pt2, (0, 220, 255), thickness, cv2.LINE_AA)
        
        # Draw players
        for track_id, point in player_points.items():
            if point is None:
                continue
            cx = max(0, min(point[0], self.width - 1))
            cy = max(0, min(point[1], self.height - 1))
            
            color = player_colors.get(track_id, (255, 255, 255))
            
            # Subtle drop shadow
            cv2.circle(pitch, (cx + 1, cy + 1), 8, (10, 10, 10), -1, lineType=cv2.LINE_AA)
            # Player dot
            cv2.circle(pitch, (cx, cy), 7, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(pitch, (cx, cy), 7, (20, 20, 20), 1, lineType=cv2.LINE_AA)
            
            # Label background & text
            label = str(track_id)
            cv2.putText(pitch, label, (cx - 7, cy - 9), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(pitch, label, (cx - 7, cy - 9), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                        
        # Draw ball
        if ball_point is not None:
            bx = max(0, min(ball_point[0], self.width - 1))
            by = max(0, min(ball_point[1], self.height - 1))
            # Glowing outer ring
            cv2.circle(pitch, (bx, by), 7, (0, 200, 255), 1, lineType=cv2.LINE_AA)
            # Core yellow dot
            cv2.circle(pitch, (bx, by), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(pitch, (bx, by), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            
        # Draw tactical overlay header / watermark
        cv2.putText(pitch, "2D TACTICAL MINIMAP", (10, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        return pitch
