# import required packages
import cv2
import time
import numpy as np
# import os
# import picamera
# from picamera import PiCamera
# from picamera.array import PiRGBArray

# import io
# import traceback
# import sys

from model import YOLO

#from safe_region import check_object_in_rectangular_area, draw_rectangular_safe_regions


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


    # define model
    yolo = YOLO(tiny=True)

    # read the classes and asign colors
    classes, COLORS = yolo.classes_colors()

    video_input = 0
    cam = cv2.VideoCapture(video_input) # detect from webcam

    res = set_res(cam, 680,480)

    print(f'Resolution: {res}')



    ret, frame = cam.read()
    # camera = PiCamera()
    # h = 480
    # w = 640
    # camera.resolution = (640, 480)
    # camera.framerate = 2
    # camera.rotation = 180
    # rawCapture = PiRGBArray(camera)
    time.sleep(1.1)
    
    # camera.capture(rawCapture, format='bgr')
    # frame = rawCapture.array
    # rawCapture.truncate(0)

    cv2.destroyAllWindows()
    

    # for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    while True:
        ret, frame = cam.read()
        if not ret:
           print("failed to grab frame")
           break
        
        #time.sleep(0.5)
        #camera.capture(rawCapture, format='bgr')
        #frame = rawCapture.array
        #time.sleep(0.5)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        detected_objects = []
        
        # frame2 = frame.array
        
        print(frame.shape)

        frame2, blob =  yolo.prepare_yolo_input(frame)

        image, detected_objects = yolo.model_inference(frame2, classes, COLORS, track_only, draw_box=True, min_confidence=0.3)

        # display output image    
        cv2.imshow("object detection", frame2)
        
        # rawCapture.truncate(0)


    cam.release()
    cv2.destroyAllWindows()
