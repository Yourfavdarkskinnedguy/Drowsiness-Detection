from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import pygame

model_filepath= os.path.join(os.getcwd(), 'best (3).pt')
model= YOLO(model_filepath)

cap= cv2.VideoCapture(1)
drowsy_frame_filepath= os.path.join(os.getcwd(), 'alert.png')
drowsy_frame= cv2.imread(drowsy_frame_filepath)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break


    result= model.predict(frame)

    for r in result:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w= x2-x1
            h= y2-y1
            label = model.names[int(box.cls[0])]

            if label == 'Drowsy':
                pygame.mixer.init() 
                alert_sound_filepath= os.path.join(os.getcwd(), 'buzzer-4-183895.mp3')
                alert_sound = pygame.mixer.Sound(alert_sound_filepath)
                alert_sound.play()
                x_offset=y_offset=40
                frame[y_offset:y_offset+drowsy_frame.shape[0], x_offset:x_offset+drowsy_frame.shape[1]] = drowsy_frame
                
                
                    #put the buzzer sound here

                print(f'drowsy attained!')
                print(f'drowsy frame attained is {frame.shape}')

            elif label == 'Alert':
                print(f'Alert attained!')
                frame = frame
                print(f'Alert frame attained is {frame}')
            
            else:
                print(f'errorrrrrrrrrrrrr!!!!!!')



            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 190), 2)
            cv2.putText(frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    print(f'shape of error image is {drowsy_frame.shape}')
    print(f'shape of frame image is {frame.shape}')
    cv2.imshow('Webcam Feed', frame)
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break