# OpenCV camera capture
# Double For Loop
import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FPS counter
start_time = time.time()
frame_counter = 0
fps = 0

while True:
    start_frame_time = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Get image dimensions
    height, width, channels = frame.shape
    # Set initial values for brightest pixel
    maxVal = 0
    maxLoc = (0,0)
    # Set inital values for reddest pixel
    maxValRed = 0
    maxLocRed = (0,0)
    # Loop through all pixels
    for x in range(width):
        for y in range(height):
            # Get pixel value
            pixel = frame[y,x]
            # Calculate brightness
            brightness = pixel[0] + pixel[1] + pixel[2]
            # Check if brightest pixel
            if brightness > maxVal:
                maxVal = brightness
                maxLoc = (x,y)
            # Calculate redness
            redness = pixel[2] - pixel[0] - pixel[1]
            # Check if reddest pixel
            if redness > maxValRed:
                maxValRed = redness
                maxLocRed = (x,y)
    # Draw circle around reddest pixel
    cv2.circle(frame, maxLocRed, 25, (0,0,255), 2)

    # Draw circle around brightest pixel
    cv2.circle(frame, maxLoc, 25, (255,0,0), 2)

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