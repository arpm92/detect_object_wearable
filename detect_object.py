# import required packages
import cv2
import numpy as np
import os

from model import YOLO

from safe_region import check_object_in_rectangular_area, draw_rectangular_safe_regions


track_only = ['person',
                'bicycle',
                'car',
                'motorbike',
                'bus',
                'truck',
                'cat',
                'dog']




#############

if __name__ == "__main__":


    # define model
    yolo = YOLO()

    # read the classes and asign colors
    classes, COLORS = yolo.classes_colors()

    video_input = 1
    cam = cv2.VideoCapture(video_input) # detect from webcam
    ret, frame = cam.read()

    
    # frame_path = 'test_img.png'
    # ret = True
    # frame = cv2.imread(frame_path)

    #warning_zone = 0.5
    warning_color = (0,255,255)
    roi_warning = cv2.selectROI(frame)
    roi_warning_zone = [roi_warning[0],roi_warning[1],(roi_warning[2] + roi_warning[0]), (roi_warning[3] + roi_warning[1])]

    #danger_zone = 0.3
    danger_color = (0,0,255)
    roi_danger = cv2.selectROI(frame)
    roi_danger_zone = [roi_danger[0],roi_danger[1],(roi_danger[2] + roi_danger[0]), (roi_danger[3] + roi_danger[1])]

    cv2.destroyAllWindows()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        detected_objects = []


        frame, blob =  yolo.prepare_yolo_input(frame)

        image, detected_objects = yolo.model_inference(frame.copy(), classes, COLORS, track_only, draw_box=True)

        image = draw_rectangular_safe_regions(image,roi_danger_zone,danger_color)

        image = draw_rectangular_safe_regions(image,roi_warning_zone,warning_color)

        for object in detected_objects:

                if check_object_in_rectangular_area(roi_warning_zone,object):
                    cv2.circle(image,(object['x'],object['y']),10,warning_color,-1)

                if check_object_in_rectangular_area(roi_danger_zone,object):
                    cv2.circle(image,(object['x'],object['y']),10,danger_color,-1)

        # display output image    
        cv2.imshow("object detection", image)


    cam.release()
    cv2.destroyAllWindows()