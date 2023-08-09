# import required packages
import cv2
import time
import numpy as np
# import os
import picamera
from picamera import PiCamera
from picamera.array import PiRGBArray
import argparse

from datetime import datetime
datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# import io
# import traceback
# import sys

from model import YOLO

import pygame

output_path = '/home/pi/detect_object_wearable/log'

def custom_print(message_to_print, log_file=output_path + '/output.txt'):
    print(message_to_print)
    with open(log_file, 'a') as of:
        of.write(message_to_print + '\n')

display = True
draw_box = True

track_only = ['person',
                'bicycle',
                'car',
                'motorbike',
                'bus',
                'truck',
                'cat',
                'dog']


def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


#############

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", "-d",  help="display True or False",default=False)

    args = parser.parse_args()

    display = args.display
    
    if display == "true" or display == "True" or display == '1':
        display = True
    else:
        display = False
        
    print(type(display),display)

    # Set sounds
    pygame.init()
    pygame.mixer.init()
    sounda= pygame.mixer.Sound("beep-sound.wav")


    # define model
    yolo = YOLO(tiny=True)

    # read the classes and asign colors
    classes, COLORS = yolo.classes_colors()

    # video_input = 0
    # cam = cv2.VideoCapture(video_input) # detect from webcam

    # res = set_res(cam, 680,480)

    # print(f'Resolution: {res}')



    # ret, frame = cam.read()
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # allow the camera to warmup
    time.sleep(0.1)

    #cv2.destroyAllWindows()
    

    # while True:
    #     ret, frame = cam.read()
    #     if not ret:
    #        print("failed to grab frame")
    #        break

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        
        frame = frame.array

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        detected_objects = []
        
        # print(frame.shape)

        frame2, blob =  yolo.prepare_yolo_input(frame)

        image, detected_objects = yolo.model_inference(frame2, classes, COLORS, track_only, draw_box=draw_box, min_confidence=0.1)

        for detected_object in detected_objects:
            detected_object['time'] = str(datetime.now())
            custom_print(str(detected_object))

            #if detected_object['ID'] == 'person' and detected_object['dist'] < 200:
            if detected_object['ID'] == 'person':
                sounda.play()
                

        if display == True:
            # display output image    
            cv2.imshow("object detection", frame2)
        
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)


    #cam.release()
    cv2.destroyAllWindows()
