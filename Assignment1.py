# OpenCV camera capture
import numpy as np
import cv2
import time

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FPS counter
start_time = time.time()
frame_counter = 0
fps = 0

while True:
    # Measure Frame time
    start_frame_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()

    # get grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur image to reduce noise a bit
    blur = cv2.blur(gray, (5,5), 0)

    #Get brightest pixel
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)

    #Draw circle around brightest pixel
    cv2.circle(frame, maxLoc, 25, (255,0,0), 2)

    
    # Find reddest pixel
    # This is done by subtracting the blue and green channels from the red channel
    # The reddest pixel will have the highest value after subtraction
    
    # Split into channels
    (B,G,R) = cv2.split(frame)
    # Calculate difference between channels
    RG = cv2.subtract(R,G)
    RB = cv2.subtract(R,B)
    # Add differences together
    added = cv2.add(RG,RB)
    # Blur image    
    blur = cv2.blur(added, (25,25), 0)
    # Get reddest pixel
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)

    # Draw circle around reddest pixel
    cv2.circle(frame, maxLoc, 25, (0,0,255), 2)

    # Calculate FPS
    frame_counter += 1
    if frame_counter >= 10:
        fps = frame_counter / (time.time() - start_time)
        start_time = time.time()
        frame_counter = 0
    
    cv2.putText(frame, f"FPS: {fps:.2f}", (1, 31), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Measure frame time
    end_frame_time = time.time()
    print(f"Frame time: {time.time() - start_frame_time:.4f}")
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    #time taken to print
    end_frame_time = time.time()
    print(f"Frame time with image: {end_frame_time - start_frame_time:.4f}")

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break