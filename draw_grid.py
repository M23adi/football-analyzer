import cv2

img = cv2.imread('first_frame.jpg')
h, w, _ = img.shape

# Draw vertical lines
for x in range(0, w, 50):
    cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
    cv2.putText(img, str(x), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Draw horizontal lines
for y in range(0, h, 50):
    cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
    cv2.putText(img, str(y), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

cv2.imwrite('grid_frame.jpg', img)
print("Grid drawn and saved to grid_frame.jpg")
