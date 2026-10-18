"""
Two-Window Calibration Tool

How it works:
  1. Two windows open: the VIDEO FRAME and a PITCH DIAGRAM
  2. Click a visible pitch marking in the VIDEO (e.g., a line intersection)
  3. Then click the SAME point on the PITCH DIAGRAM
  4. Repeat for 4 pairs total
  5. Press any key when done

Use clearly identifiable points: line intersections, arc-to-line junctions, the center spot, etc.
Right-click on either window to UNDO the last point.
"""
import cv2
import json
import numpy as np

# ---- State ----
video_points = []
pitch_points = []
clicking_on = "video"  # alternates: "video" -> "pitch" -> "video" ...
pair_count = 0

# ---- Build a pitch diagram (105m x 68m, scaled to 700x453) ----
SCALE = 6.67  # px per meter
P_W = int(105 * SCALE)  # ~700
P_H = int(68 * SCALE)   # ~453

def build_pitch():
    pitch = np.full((P_H, P_W, 3), (34, 139, 34), dtype=np.uint8)
    c = (255, 255, 255)
    t = 2
    def m(x, y):
        return (int(x * SCALE), int(y * SCALE))
    
    cv2.rectangle(pitch, m(0, 0), m(105, 68), c, t)
    cv2.line(pitch, m(52.5, 0), m(52.5, 68), c, t)
    cv2.circle(pitch, m(52.5, 34), int(9.15 * SCALE), c, t)
    cv2.circle(pitch, m(52.5, 34), 4, c, -1)
    
    pen_top = (68 - 40.32) / 2
    pen_bot = pen_top + 40.32
    goal_top = (68 - 18.32) / 2
    goal_bot = goal_top + 18.32
    
    cv2.rectangle(pitch, m(0, pen_top), m(16.5, pen_bot), c, t)
    cv2.rectangle(pitch, m(0, goal_top), m(5.5, goal_bot), c, t)
    cv2.circle(pitch, m(11, 34), 3, c, -1)
    cv2.ellipse(pitch, m(11, 34), (int(9.15 * SCALE), int(9.15 * SCALE)), 0, -53, 53, c, t)
    
    cv2.rectangle(pitch, m(105 - 16.5, pen_top), m(105, pen_bot), c, t)
    cv2.rectangle(pitch, m(105 - 5.5, goal_top), m(105, goal_bot), c, t)
    cv2.circle(pitch, m(94, 34), 3, c, -1)
    cv2.ellipse(pitch, m(94, 34), (int(9.15 * SCALE), int(9.15 * SCALE)), 0, 127, 233, c, t)
    
    return pitch

# ---- Click handlers ----
def video_click(event, x, y, flags, params):
    global clicking_on, pair_count
    
    if event == cv2.EVENT_RBUTTONDOWN:
        undo(params)
        return
    
    if event == cv2.EVENT_LBUTTONDOWN and clicking_on == "video" and pair_count < 4:
        video_points.append((x, y))
        idx = len(video_points)
        cv2.circle(params['vimg'], (x, y), 8, (0, 0, 255), -1)
        cv2.putText(params['vimg'], str(idx), (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        update_status(params, f"Now click the SAME point on the PITCH DIAGRAM (point {idx})")
        clicking_on = "pitch"
        cv2.imshow('Video Frame', params['vimg'])

def pitch_click(event, x, y, flags, params):
    global clicking_on, pair_count
    
    if event == cv2.EVENT_RBUTTONDOWN:
        undo(params)
        return
    
    if event == cv2.EVENT_LBUTTONDOWN and clicking_on == "pitch" and pair_count < 4:
        pitch_points.append((x, y))
        idx = len(pitch_points)
        cv2.circle(params['pimg'], (x, y), 8, (0, 0, 255), -1)
        cv2.putText(params['pimg'], str(idx), (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        pair_count += 1
        
        if pair_count < 4:
            update_status(params, f"Pair {pair_count}/4 done. Click point {pair_count+1} on the VIDEO")
            clicking_on = "video"
        else:
            update_status(params, "All 4 pairs done! Press any key to SAVE and close.")
        
        cv2.imshow('Pitch Diagram', params['pimg'])

def undo(params):
    global clicking_on, pair_count
    if clicking_on == "pitch" and len(video_points) > len(pitch_points):
        video_points.pop()
        clicking_on = "video"
        # Redraw video
        params['vimg'][:] = params['vimg_clean'].copy()
        for i, pt in enumerate(video_points):
            cv2.circle(params['vimg'], pt, 8, (0, 0, 255), -1)
            cv2.putText(params['vimg'], str(i+1), (pt[0]+10, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        update_status(params, f"Undone. Click point {len(video_points)+1} on the VIDEO")
        cv2.imshow('Video Frame', params['vimg'])
    elif clicking_on == "video" and len(pitch_points) > 0:
        pitch_points.pop()
        video_points.pop()
        pair_count -= 1
        # Redraw pitch
        params['pimg'][:] = params['pimg_clean'].copy()
        for i, pt in enumerate(pitch_points):
            cv2.circle(params['pimg'], pt, 8, (0, 0, 255), -1)
            cv2.putText(params['pimg'], str(i+1), (pt[0]+10, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Redraw video
        params['vimg'][:] = params['vimg_clean'].copy()
        for i, pt in enumerate(video_points):
            cv2.circle(params['vimg'], pt, 8, (0, 0, 255), -1)
            cv2.putText(params['vimg'], str(i+1), (pt[0]+10, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        update_status(params, f"Undone. Click point {pair_count+1} on the VIDEO")
        cv2.imshow('Video Frame', params['vimg'])
        cv2.imshow('Pitch Diagram', params['pimg'])

def update_status(params, text):
    # Update status bar on both windows
    for key, win in [('vimg', 'Video Frame'), ('pimg', 'Pitch Diagram')]:
        img = params[key]
        cv2.rectangle(img, (0, 0), (img.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, img)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Two-Window Football Pitch Homography Calibration Tool")
    parser.add_argument("--video", type=str, default="data/test_match.mp4", help="Path to video file")
    parser.add_argument("--output", type=str, default="calibration.json", help="Path to output calibration json")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ret, vframe = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read video at {args.video}")
        return
    
    pitch_diagram = build_pitch()
    
    vimg = vframe.copy()
    pimg = pitch_diagram.copy()
    
    params = {
        'vimg': vimg, 'vimg_clean': vframe.copy(),
        'pimg': pimg, 'pimg_clean': pitch_diagram.copy()
    }
    
    print("=" * 65)
    print("TWO-WINDOW CALIBRATION")
    print("  1. Click a line intersection on the VIDEO")
    print("  2. Click the SAME point on the PITCH DIAGRAM")
    print("  3. Repeat 4 times. Right-click to undo.")
    print("  4. Press any key when done.")
    print("=" * 65)
    
    cv2.namedWindow('Video Frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Pitch Diagram', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Frame', 960, 540)
    cv2.resizeWindow('Pitch Diagram', 700, 453)
    
    update_status(params, "Click point 1 on the VIDEO")
    
    cv2.setMouseCallback('Video Frame', video_click, params)
    cv2.setMouseCallback('Pitch Diagram', pitch_click, params)
    
    cv2.imshow('Video Frame', vimg)
    cv2.imshow('Pitch Diagram', pimg)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if pair_count == 4:
        # Convert pitch diagram pixel coords to real-world meters, then back to minimap coords
        # The pitch diagram was drawn at SCALE px/m, so divide by SCALE to get meters.
        pitch_real = [(x / SCALE, y / SCALE) for x, y in pitch_points]
        
        config = {
            "src_points": video_points,
            "dst_points_meters": pitch_real,
            "pitch_diagram_scale": SCALE,
        }
        with open(args.output, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nSaved 4 point pairs to {args.output}!")
        for i in range(4):
            print(f"  Pair {i+1}: video{video_points[i]} -> pitch({pitch_real[i][0]:.1f}m, {pitch_real[i][1]:.1f}m)")
    else:
        print(f"\nError: Need 4 pairs, got {pair_count}. Please try again.")

if __name__ == "__main__":
    main()
