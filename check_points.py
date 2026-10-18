import cv2

img = cv2.imread('first_frame.jpg')

points = [
    (0, 285),     # Top-Left
    (310, 410),   # Top-Right
    (0, 1030),    # Bottom-Left
    (145, 890)    # Bottom-Right
]

for i, p in enumerate(points):
    cv2.circle(img, p, 5, (0, 0, 255), -1)
    cv2.putText(img, str(i+1), (p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite('check_points.jpg', img)
print("Points drawn and saved to check_points.jpg")
