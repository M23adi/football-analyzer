import argparse
import json
import os
import time
import cv2
import numpy as np

from src.detection import MatchAnalyzer
from src.ball_tracker import BallTracker
from src.team_assigner import TeamAssigner
from src.camera_transformation import CameraTransformation
from src.minimap import Minimap

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tactical Football Match Analyzer: AI Player Tracking, Jersey Clustering & 2D Minimap"
    )
    parser.add_argument(
        "--video", type=str, default="data/test_match.mp4",
        help="Path to input football match video (default: data/test_match.mp4)"
    )
    parser.add_argument(
        "--calibration", type=str, default="calibration.json",
        help="Path to calibration.json mapping video coordinates to 2D pitch (default: calibration.json)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to export analyzed side-by-side video (e.g. output_tactical.mp4)"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8m.pt",
        help="YOLO model checkpoint or file path (default: yolov8m.pt)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Computation device ('cuda', 'cpu', or None for auto-detect)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.2,
        help="YOLO detection confidence threshold (default: 0.2)"
    )
    parser.add_argument(
        "--scale", type=float, default=6.5,
        help="Minimap scale in pixels per meter (default: 6.5)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run in headless mode without GUI window (recommended for servers/Docker)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of frames to process (default: all)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Validate inputs
    if not os.path.exists(args.video):
        print(f"[Error] Video file not found: {args.video}")
        return

    if not os.path.exists(args.calibration):
        print(f"[Error] Calibration file not found: {args.calibration}")
        print("Please run 'python calibrate.py' first to generate calibration.json.")
        return

    with open(args.calibration, 'r') as f:
        cal = json.load(f)
    src_points = cal['src_points']
    dst_points_m = cal['dst_points_meters']

    # 2. Minimap scale and dimensions (105m x 68m standard pitch)
    SCALE = args.scale
    PITCH_W = int(round(105 * SCALE))   # e.g., 682px at 6.5
    PITCH_H = int(round(68 * SCALE))    # e.g., 442px at 6.5
    dst_points = [(int(round(x * SCALE)), int(round(y * SCALE))) for x, y in dst_points_m]

    # 3. Initialize CV modules
    analyzer = MatchAnalyzer(model_path=args.model, device=args.device, conf=args.conf)
    ball_memory = BallTracker(max_missing_frames=12)
    team_assigner = TeamAssigner()
    camera_transform = CameraTransformation(src_points, dst_points)
    minimap = Minimap(width=PITCH_W, height=PITCH_H)

    # 4. Setup Video Capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Error] Could not open video at: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video Info] Source: {args.video} | FPS: {fps:.1f} | Frames: {total_frames}")

    # Video Writer setup (if --output specified)
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        print(f"[Output] Saving analyzed video to: {args.output}")

    # Tracking state
    raw_player_colors = {}
    player_colors = {}
    teams_classified = False

    frame_count = 0
    t0 = time.time()
    fps_display = 0.0

    print("Starting tactical match analysis... (Press 'q' in window to stop)")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                print(f"[Info] Reached max frames limit ({args.max_frames}). Stopping.")
                break

            loop_start = time.time()

            # Step A: Detection & Tracking
            results = analyzer.process_frame(frame)

            # Step B: Logic & Homography Projections
            current_ball_box = None
            player_points_2d = {}

            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    box_height = y2 - y1

                    # Filter on-pitch players (reject crowd or tiny background boxes)
                    if int(cls) == 0 and y2 > 250 and box_height > 30:
                        if track_id not in raw_player_colors:
                            color = team_assigner.get_player_color(frame, box)
                            raw_player_colors[track_id] = color

                        # Player feet coordinate (bottom-center)
                        feet_coord = (int((x1 + x2) / 2), y2)
                        pt_2d = camera_transform.transform_point(feet_coord)
                        if pt_2d:
                            player_points_2d[track_id] = pt_2d

                    # Ball detection
                    elif int(cls) == 32:
                        current_ball_box = box

            # Step C: Team Classification
            if not teams_classified and len(raw_player_colors) >= 8:
                player_colors = team_assigner.fit_teams(raw_player_colors)
                teams_classified = True
            elif teams_classified:
                for pid, raw_col in raw_player_colors.items():
                    if pid not in player_colors:
                        player_colors[pid] = team_assigner.assign_player(raw_col)

            # Step D: Ball Temporal Memory & Extrapolation
            smoothed_ball = ball_memory.update(current_ball_box)

            ball_point_2d = None
            if smoothed_ball is not None:
                bx1, by1, bx2, by2 = map(int, smoothed_ball)
                ball_center = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
                ball_point_2d = camera_transform.transform_point(ball_center)

            # Transform ball trajectory history to 2D minimap
            ball_history_2d = []
            for hist_pt in ball_memory.get_trajectory():
                pt2d = camera_transform.transform_point(hist_pt)
                if pt2d:
                    ball_history_2d.append(pt2d)

            # Step E: Visualization
            annotated_frame = analyzer.draw_detections(frame, results, player_colors, smoothed_ball)

            # Calculate FPS
            loop_elapsed = time.time() - loop_start
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(loop_elapsed, 1e-4))

            # HUD overlay on broadcast frame
            hud_text = f"Frame: {frame_count}/{total_frames} | FPS: {fps_display:.1f}"
            cv2.putText(annotated_frame, hud_text, (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated_frame, hud_text, (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

            # Resize broadcast frame to match minimap height
            target_h = PITCH_H
            aspect = annotated_frame.shape[1] / annotated_frame.shape[0]
            target_w = int(round(target_h * aspect))
            resized_frame = cv2.resize(annotated_frame, (target_w, target_h))

            # Render Minimap with ball trajectory
            minimap_frame = minimap.draw_points(
                player_points_2d,
                ball_point_2d,
                player_colors,
                ball_history_2d=ball_history_2d
            )

            # Ensure matching height for horizontal concatenation
            if resized_frame.shape[0] != minimap_frame.shape[0]:
                minimap_frame = cv2.resize(minimap_frame, (minimap_frame.shape[1], resized_frame.shape[0]))

            # Combine Side-by-Side: Broadcast View | 2D Minimap View
            combined_frame = cv2.hconcat([resized_frame, minimap_frame])

            # Initialize VideoWriter if requested
            if args.output and writer is None:
                comb_h, comb_w = combined_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.output, fourcc, fps, (comb_w, comb_h))

            if writer is not None:
                writer.write(combined_frame)

            # Display window unless headless
            if not args.no_display:
                cv2.imshow("Tactical Football Match Analyzer", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[User] Analysis interrupted by user.")
                    break

            if frame_count % 30 == 0:
                elapsed_total = time.time() - t0
                overall_fps = frame_count / max(elapsed_total, 1e-4)
                print(f"[Progress] Frame {frame_count}/{total_frames} ({overall_fps:.1f} FPS) | Players: {len(player_points_2d)}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[Success] Video saved successfully to: {args.output}")
        if not args.no_display:
            cv2.destroyAllWindows()

    total_time = time.time() - t0
    avg_fps = frame_count / max(total_time, 1e-4)
    print(f"\n[Completed] Processed {frame_count} frames in {total_time:.2f}s ({avg_fps:.1f} FPS average).")

if __name__ == "__main__":
    main()