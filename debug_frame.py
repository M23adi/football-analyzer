"""Debug v3: generate debug frame with current calibration."""
import cv2
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from src.detection import MatchAnalyzer
from src.ball_tracker import BallTracker
from src.team_assigner import TeamAssigner
from src.camera_transformation import CameraTransformation
from src.minimap import Minimap

def main():
    analyzer = MatchAnalyzer()
    ball_memory = BallTracker(max_missing_frames=7)
    team_assigner = TeamAssigner()

    with open('calibration.json', 'r') as f:
        cal = json.load(f)
    src_points = cal['src_points']
    
    SCALE = 6.5
    PITCH_W = int(105 * SCALE)
    PITCH_H = int(68 * SCALE)
    
    dst_points_m = cal['dst_points_meters']
    dst_points = [(int(x * SCALE), int(y * SCALE)) for x, y in dst_points_m]

    camera_transform = CameraTransformation(src_points, dst_points)
    minimap = Minimap(width=PITCH_W, height=PITCH_H)

    cap = cv2.VideoCapture('data/test_match.mp4')
    
    raw_player_colors = {}
    TEAM_COLORS = [(0, 0, 230), (0, 230, 230)]
    
    # Process 60 frames
    for frame_num in range(60):
        ret, frame = cap.read()
        if not ret:
            break
        results = analyzer.process_frame(frame)
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            for box, track_id, cls in zip(boxes, track_ids, classes):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                if int(cls) == 0 and y2 > 250 and track_id not in raw_player_colors:
                    color = team_assigner.get_player_color(frame, box)
                    raw_player_colors[track_id] = color

    # Analyze raw colors
    print(f"Total players detected: {len(raw_player_colors)}")
    print("\nRaw jersey colors (BGR):")
    for pid, col in sorted(raw_player_colors.items()):
        brightness = sum(col) / 3
        print(f"  ID={pid}: BGR={col}, brightness={brightness:.0f}")
    
    # Cluster into 3 groups (Team A, Team B, Referee/Other)
    ids = list(raw_player_colors.keys())
    colors_array = np.array([raw_player_colors[pid] for pid in ids], dtype=np.float32)
    km = KMeans(n_clusters=3, n_init='auto', random_state=42)
    km.fit(colors_array)
    
    print("\nCluster centers (BGR):")
    for i, center in enumerate(km.cluster_centers_):
        brightness = sum(center) / 3
        count = sum(1 for l in km.labels_ if l == i)
        print(f"  Cluster {i}: BGR=({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f}), brightness={brightness:.0f}, count={count}")
    
    # Build the last frame output
    player_colors = {}
    player_points_2d = {}
    current_ball_box = None
    
    # Identify the darkest cluster as referee
    brightnesses = [sum(c) / 3 for c in km.cluster_centers_]
    ref_cluster = np.argmin(brightnesses)
    team_clusters = [i for i in range(3) if i != ref_cluster]
    
    for pid, label in zip(ids, km.labels_):
        if label == ref_cluster:
            player_colors[pid] = (128, 128, 128)  # Grey for referee
        elif label == team_clusters[0]:
            player_colors[pid] = TEAM_COLORS[0]
        else:
            player_colors[pid] = TEAM_COLORS[1]
    
    # Process last frame
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        for box, track_id, cls in zip(boxes, track_ids, classes):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            if int(cls) == 0 and y2 > 250:
                feet_coord = (int((x1 + x2) / 2), y2)
                pt_2d = camera_transform.transform_point(feet_coord)
                if pt_2d:
                    player_points_2d[track_id] = pt_2d
            elif int(cls) == 32:
                current_ball_box = box
    
    smoothed_ball = ball_memory.update(current_ball_box)
    ball_point_2d = None
    if smoothed_ball is not None:
        bx1, by1, bx2, by2 = map(int, smoothed_ball)
        ball_center = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
        ball_point_2d = camera_transform.transform_point(ball_center)
    
    output_frame = analyzer.draw_detections(frame, results, player_colors, smoothed_ball)
    minimap_frame = minimap.draw_points(player_points_2d, ball_point_2d, player_colors)
    
    target_height = PITCH_H
    aspect_ratio = output_frame.shape[1] / output_frame.shape[0]
    target_width = int(target_height * aspect_ratio)
    resized_frame = cv2.resize(output_frame, (target_width, target_height))
    
    if resized_frame.shape[0] != minimap_frame.shape[0]:
        minimap_frame = cv2.resize(minimap_frame, (minimap_frame.shape[1], resized_frame.shape[0]))
    
    combined = cv2.hconcat([resized_frame, minimap_frame])
    cv2.imwrite('debug_output_v3.jpg', combined)
    print(f"\nSaved debug_output_v3.jpg ({combined.shape[1]}x{combined.shape[0]})")
    
    cap.release()

if __name__ == "__main__":
    main()
