from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
import math

IN1=17
IN2=27
IN3=22
IN4=23
ENA=18
ENB=24
dutyleft= 40
dutyright=40
x=320
y=0
w=0
h=0
kernal=np.ones((3,3),np.uint8)

cv2.namedWindow("normal")

def on_trackbar(val):
    global threshold_value
    threshold_value = val

threshold_value = 101  # Default threshold value

#flags
shape_count=0
arrow_count=0

# Create trackbar for threshold adjustment
cv2.createTrackbar("Threshold", "normal", threshold_value, 255, on_trackbar)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

pwmA = GPIO.PWM(ENA,200)
pwmB = GPIO.PWM(ENB,200)
pwmA.start(dutyleft)
pwmB.start(dutyright)

def moveforward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def movebackward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    
def turnleft():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def turnright():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def stopcar():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    
def detect_shape(cnt):
    shape = "unknown"
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    area = cv2.contourArea(cnt)
    yep='unknown'

    if len(approx) == 3:
        shape = "triangle"
        yep='shape'
    elif len(approx) == 5:
        shape = "pentagon"
        yep='shape'
    elif len(approx) == 6:
        shape = "hexagon"
        yep='shape'
    elif len(approx) == 7:  # Assuming arrows have 7 vertices in your application
        # Additional geometric checks can go here
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        rect = cv2.minAreaRect(cnt)
        max_distance = 0
        tip_index = 0
        for i in range(len(approx)):
            dist = np.linalg.norm(approx[i][0] - (cX, cY))
            if dist > max_distance:
                max_distance = dist
                tip_index = i

        # Calculate the angle between the centroid and the tip of the arrow with respect to a reference axis
        ref_angle = 90  # Angle of the reference axis (horizontal line)
        tip_angle = math.atan2(cY - approx[tip_index][0][1], approx[tip_index][0][0] - cX) * 180 / math.pi
        arrow_angle = tip_angle - ref_angle
        arrow_angle %= 360  # Ensure the angle is within [0, 360) range

        # Determine the direction based on the arrow angle
        if 45 <= arrow_angle < 135:
            shape = "arrow (right)"
            yep='arrow'
        elif 135 <= arrow_angle < 225:
            shape = "arrow (down)"
            yep='arrow'
        elif 225 <= arrow_angle < 315:
            shape = "arrow (left)"
            yep='arrow'
        else:
            shape = "arrow (up)"
            yep='arrow'
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.90 <= ar <= 1.1 else "rectangle"
        yep='shape'
    else:
        # Calculating the circularity using the formula (4π*Area)/(Perimeter^2)
        circularity = (4 * math.pi * area) / (peri * peri)
        
        # Adjust the circularity threshold to improve differentiation
        # A perfect circle would have a circularity of 1, but due to pixelation and approximation, it will rarely be exactly 1
        if circularity > 0.80:  # Adjust this threshold as needed
            shape = "full circle"
            yep='shape'
        elif circularity < 0.7:
            # Additional checks for "partial circle" or other shapes can be included here
            shape = "partial circle"
            yep='shape'
    return shape,yep

camera=PiCamera()
camera.resolution=(640,368)
rawCapture=PiRGBArray(camera,size=(640,368))
time.sleep(0.1)

try:
    for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        image=frame.array
        roi=image[230:360,0:639]
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        HSVimage=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        #HSVimage=cv2.GaussianBlur(HSVimage,(3,3),0)
        Blackline=cv2.inRange(roi,(0,0,0),(60,60,60))
        Redline=cv2.inRange(HSVimage,(130,80,80),(180,255,255))
        Redline2=cv2.inRange(HSVimage,(0,80,70),(15,255,255))
        Greenline=cv2.inRange(HSVimage,(50,150,70),(90,255,155))
        Blueline=cv2.inRange(HSVimage,(90,80,2),(135,255,255))
        Yellowline=cv2.inRange(HSVimage,(20,100,80),(35,255,155))
        Redline=cv2.erode(Redline,kernal,iterations=2)
        Greenline=cv2.erode(Greenline,kernal,iterations=2)
        Blueline=cv2.erode(Blueline,kernal,iterations=2)
        Yellowline=cv2.erode(Yellowline,kernal,iterations=2)
        Blackline=cv2.erode(Blackline,kernal,iterations=2)
        Redline2=cv2.erode(Redline2,kernal,iterations=2)
        Redline=cv2.bitwise_or(Redline,Redline2,2,mask=None)
        blackcontours, blackhierarchy = cv2.findContours(Blackline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        redcontours, redhierarchy = cv2.findContours(Redline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        greencontours, greenhierarchy = cv2.findContours(Greenline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bluecontours, bluehierarchy = cv2.findContours(Blueline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        yellowcontours, yellowhierarchy = cv2.findContours(Yellowline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #if len(redcontours)>0:
            #x,y,w,h=cv2.boundingRect(redcontours[0])
        #elif len(greencontours)>0:
            #x,y,w,h=cv2.boundingRect(greencontours[0])
        if len(bluecontours)>0:
            x,y,w,h=cv2.boundingRect(bluecontours[0])
        elif len(yellowcontours)>0:
            x,y,w,h=cv2.boundingRect(yellowcontours[0])
        elif len(blackcontours)>0:
            x,y,w,h=cv2.boundingRect(blackcontours[0])
        error=int(x+(w/2))-320
        if len(redcontours)==0 and len(greencontours)==0 and len(bluecontours)==0 and len(yellowcontours)==0 and len(blackcontours)==0:
            error=0
        #print("error:",error)
        #print("red",len(redcontours))
        #print("green",len(greencontours))
        #print("blue",len(bluecontours))
        #print("yellow",len(yellowcontours))
        #print("black",len(blackcontours))
        cv2.imshow("red",Redline)
        cv2.imshow("green",Greenline)
        cv2.imshow("blue",Blueline)
        cv2.imshow("yellow",Yellowline)
        cv2.imshow("black",Blackline)
        rawCapture.truncate(0)
        key= cv2.waitKey(1) & 0xFF
        if key==ord("q"):
            cv2.destroyAllWindows()
            break
        if error<-160:
            dutyleft=46
            dutyright=46
            pwmA.ChangeDutyCycle(dutyleft)
            pwmB.ChangeDutyCycle(dutyright)
            turnleft()
        elif error>160:
            dutyleft=44
            dutyright=44
            pwmA.ChangeDutyCycle(dutyleft)
            pwmB.ChangeDutyCycle(dutyright)
            turnright()
        elif error<-60:
            dutyleft=46
            dutyright=46
            pwmA.ChangeDutyCycle(dutyleft)
            pwmB.ChangeDutyCycle(dutyright)
            turnleft()
        elif error>60:
            dutyleft=50
            dutyright=50
            pwmA.ChangeDutyCycle(dutyleft)
            pwmB.ChangeDutyCycle(dutyright)
            turnright()
        elif error==0:
            dutyleft=28
            dutyright=28
            pwmA.ChangeDutyCycle(dutyleft)
            pwmB.ChangeDutyCycle(dutyright)
            movebackward()
        else:
            dutyleft=25
            dutyright=25
            pwmA.ChangeDutyCycle(dutyleft)
            pwmB.ChangeDutyCycle(dutyright)
            moveforward()
        
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Apply binary thresholding
        _, thresh_frame = cv2.threshold(blurred_frame, threshold_value, 255, cv2.THRESH_BINARY)
        cv2.imshow("BW",thresh_frame)
        
        # Optionally, apply another round of Gaussian blur here before Canny edge detection for smoother edges
        blurred_for_edges = cv2.GaussianBlur(thresh_frame, (5, 5), 0)

        # Apply Canny edge detection on the blurred image
        edges = cv2.Canny(blurred_for_edges, 30, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


 
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 8000:  # Filter out contours with area smaller than 500 pixels
                shape,yep = detect_shape(cnt)
                
                cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    #print('x:',cX)
                    #print('y:',cY)
                    '''
                    if cY>=368:
                        cY=367
                    if cX>=368:
                        cX=367
                    '''
                    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
                    #print(thresh_frame[cX,cY])
                    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    if shape_count==0 and yep=='shape':
                        print(shape +' detected')
                        shape_count=1
                    elif arrow_count==0 and yep=='arrow':
                        print(shape+' detected')
                        arrow_count=1
                    
                    
        cv2.imshow("normal",image)
        
except KeyboardInterrupt:
    GPIO.cleanup()
GPIO.cleanup()