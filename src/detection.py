import cv2
import torch
from ultralytics import YOLO

class MatchAnalyzer:
    """
    Performs player and ball detection and tracking using YOLOv8 with ByteTrack.
    Supports auto device detection (CUDA GPU with fallback to CPU).
    """
    def __init__(self, model_path='yolov8m.pt', device=None, conf=0.2):
        self.conf = conf
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"[MatchAnalyzer] Loading model '{model_path}' on device '{self.device}'...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
    def process_frame(self, frame):
        """
        Runs YOLO tracking on a single video frame for players (class 0) and ball (class 32).
        ByteTrack persistence (persist=True) assigns consistent track IDs.
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[0, 32],
            conf=self.conf,
            verbose=False,
            device=self.device
        )
        return results[0]

    def draw_detections(self, frame, results, player_colors, ball_box):
        """
        Renders custom bounding boxes and track labels with assigned team colors.
        """
        annotated_frame = frame.copy()
        
        # Draw Players
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                
                if int(cls) == 0:  # Person / Player
                    color = player_colors.get(track_id, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw ID tag with a solid background badge for high legibility
                    label = f"ID: {track_id}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    badge_y1 = max(0, y1 - th - 6)
                    badge_y2 = y1
                    cv2.rectangle(annotated_frame, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)
                    # Use contrasting text color
                    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
                    cv2.putText(annotated_frame, label, (x1 + 3, badge_y2 - 3), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
                                
        # Draw the Ball
        if ball_box is not None:
            bx1, by1, bx2, by2 = map(int, ball_box)
            ball_color = (0, 255, 255)  # Vivid yellow
            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), ball_color, 2)
            cv2.putText(annotated_frame, "Ball", (bx1, max(15, by1 - 8)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ball_color, 2, cv2.LINE_AA)
                        
        return annotated_frame