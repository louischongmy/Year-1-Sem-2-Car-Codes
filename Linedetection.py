import cv2
import numpy as np
import RPi.GPIO as GPIO
from time import sleep

# Setup GPIO for demonstration purposes, adjust as per your setup
IN1 = 17 # RF
IN2 = 27 # RB
IN3 = 22 # LB
IN4 = 23 # LF
ENA = 18
ENB = 24

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup GPIO pins for motor driver
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# Create PWM instances
pwm_a = GPIO.PWM(ENA, 1000)  # PWM frequency = 1000Hz
pwm_b = GPIO.PWM(ENB, 1000)  # PWM frequency = 1000Hz

# Start PWM with 0% duty cycle (motors off)
pwm_a.start(0)
pwm_b.start(0)

threshold_value = 87  # Default threshold value

# Callback function for the trackbar, to update the threshold value
def on_trackbar(val):
    global threshold_value
    threshold_value = val

def forward():
    GPIO.output([IN1, IN2], [GPIO.HIGH, GPIO.LOW])
    GPIO.output([IN3, IN4], [GPIO.LOW, GPIO.HIGH])
    pwm_a.ChangeDutyCycle(40)  # Adjust duty cycle for desired speed
    pwm_b.ChangeDutyCycle(40)
                            # Adjust duty cycle for desired speed
    print("Moving forward")

# Function to drive the car backward
def backward():
    GPIO.output([IN1, IN2], [GPIO.LOW, GPIO.HIGH])
    GPIO.output([IN3, IN4], [GPIO.HIGH, GPIO.LOW])
    pwm_a.ChangeDutyCycle(35)  # Adjust duty cycle for desired speed
    pwm_b.ChangeDutyCycle(35)  # Adjust duty cycle for desired speed
    sleep(0.05)
    
def turn_left():
    GPIO.output([IN1, IN2], [GPIO.HIGH, GPIO.LOW])
    GPIO.output([IN3, IN4], [GPIO.HIGH, GPIO.LOW])
    pwm_a.ChangeDutyCycle(52)  # Adjust duty cycle for desired speed
    pwm_b.ChangeDutyCycle(52)  # Adjust duty cycle for desired speed
    print("Turning left")
    #sleep(0.08)
    
def turn_right():
    GPIO.output([IN1, IN2], [GPIO.LOW, GPIO.HIGH])
    GPIO.output([IN3, IN4], [GPIO.LOW, GPIO.HIGH])
    pwm_a.ChangeDutyCycle(57)  # Adjust duty cycle for desired speed
    pwm_b.ChangeDutyCycle(57)  # Adjust duty cycle for desired speed
    print("Turning right")
    #sleep(0.08)

# Function to stop the car
def stop():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwm_a.ChangeDutyCycle(0)  # Motors off
    pwm_b.ChangeDutyCycle(0)  # Motors off

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 500)  # Width
cap.set(4, 240)  # Height

cv2.namedWindow("Frame")

# Create trackbar for threshold adjustment
cv2.createTrackbar("Threshold", "Frame", threshold_value, 255, on_trackbar)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary threshold inversion for black line detection
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming the largest contour is the line
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            # Calculate the center of the contour
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Draw the center dot
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Decision logic for robot movement (placeholder)
            frame_center = frame.shape[1] // 2
            print('x:',cx)
            print('y:',cy)
            if cx < frame_center - 75:
                #backward()
                turn_left()
            elif cx > frame_center + 75:
                #backward()
                turn_right()
            else:
                forward()
    else:
        print("No line detected")

    cv2.imshow("Frame", frame)
    cv2.imshow("BW",thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()