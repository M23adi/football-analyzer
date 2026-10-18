import cv2

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the coordinate to the terminal in the format we need
        print(f"[{x}, {y}],")
        # Draw a red dot where you clicked so you don't lose track
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Click 4 Reference Points', img)

video_path = 'data/test_match.mp4'  
cap = cv2.VideoCapture(video_path)
ret, img = cap.read()

if ret:
    print("Window opened! Click on 4 points that form a rectangle on the pitch.")
    print("Press any key on your keyboard to close the window when you are done.")
    
    # --- NEW: Create a resizable window that fits on your screen ---
    cv2.namedWindow('Click 4 Reference Points', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Click 4 Reference Points', 1280, 720) 
    # ---------------------------------------------------------------
    
    cv2.imshow('Click 4 Reference Points', img)
    cv2.setMouseCallback('Click 4 Reference Points', click_event)
    
    # Wait for the user to press a key before closing
    cv2.waitKey(0)

cv2.destroyAllWindows()
cap.release()