import cv2
import numpy as np

cap = cv2.VideoCapture('lane1-straight.mp4')  # Replace with your video file or camera index

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Lane detection code goes here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    
    # Define points for the polygon mask
    # points = np.array([[100, frame.shape[0]], [500, 250], [800, 250], [frame.shape[1], frame.shape[0]]], np.int32)
    # points = points.reshape((-1, 1, 2))
    roi_vertices = [
    np.array([[100, 540], [900, 540], [515, 320], [450, 320]], np.int32)]

    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_image = cv2.bitwise_and(canny, mask)  # Apply the mask to the canny edge image
    
    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(masked_image, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
    # Draw detected lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Lane Detection', frame)  # Show the frame with detected lanes
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()