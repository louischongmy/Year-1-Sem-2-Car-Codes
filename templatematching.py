from picamera.array import PiRGBArray
from picamera import PiCamera
from matplotlib import pyplot as plt
import time
import cv2
import numpy as np

# Initialize the camera and grab a reference to the raw camera capture
'''
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
'''

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)

# Load the template once before the loop
template = cv2.imread('facerecognisation.png', 0)
w, h = template.shape[::-1]

# Allow the camera to warm up
#time.sleep(0.1)

# Capture frames from the camera
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
while True:
    # Grab the raw NumPy array representing the image
    #image = frame.array
    ret, image = cap.read()

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where(res >= threshold)

    # Draw a rectangle around the matches
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        cv2.putText(image,'left arrow',(200,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # Clear the stream in preparation for the next frame
    #rawCapture.truncate(0)

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()