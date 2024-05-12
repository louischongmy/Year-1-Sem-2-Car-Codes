import cv2
import numpy as np
import math

# Initialize the threshold value globally
threshold_value = 127  # Default threshold value

# Callback function for the trackbar, to update the threshold value
def on_trackbar(val):
    global threshold_value
    threshold_value = val

def detect_shape(cnt):
    shape = "unknown"
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    elif len(approx) == 6:
        shape = "hexagon"
    else:
        # Adjustments for circularity criteria for better circle detection
        area = cv2.contourArea(cnt)
        bounding_box_area = cv2.minAreaRect(cnt)[1][0] * cv2.minAreaRect(cnt)[1][1]
        circularity = (4 * math.pi * area) / (peri * peri)
        
        # Additional check using area ratio to differentiate circles from polygons
        area_ratio = area / bounding_box_area
        
        if circularity > 0.8 and area_ratio > 0.5:  # These thresholds may need fine-tuning
            shape = "full circle"
        elif circularity > 0.7 and area_ratio > 0.4:  # Adjust based on your requirements
            shape = "partial circle"
        else:
            # Handle cases where the shape is not recognized as any specific polygon or circle
            shape = f"polygon with {len(approx)} vertices"

    return shape


# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

cv2.namedWindow("Shape Detection")

# Create trackbar for threshold adjustment
cv2.createTrackbar("Threshold", "Shape Detection", threshold_value, 255, on_trackbar)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # If no frame is captured / end of video, exit the loop

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Apply binary thresholding using the trackbar value
    _, thresh_frame = cv2.threshold(blurred_frame, threshold_value, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh_frame, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:  # Filter out contours with area smaller than 500 pixels
            shape = detect_shape(cnt)
            
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



    cv2.imshow("Shape Detection", frame)
    cv2.imshow("BW", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

