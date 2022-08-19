import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time

import traceback
import logging

from yolo_class import YOLO


current_path = os.getcwd()


###########

if __name__=="__main__":

    track_only = ["person", "bicycle", "car", "motorbike"]
    yolo = YOLO(track_only=track_only)

    # set start time to current time
    start_time = time.time()
    # displays the frame rate every 2 second
    display_time = 2
    # Set primarry FPS to 0
    fps = 0

    # we create the video capture object cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("We cannot open webcam")

    while True:
        try:
            ret, frame = cap.read()
            # resize our captured frame if we need
            frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

            # # detect object on our frame
            r_image, ObjectsList = yolo.detect_img(frame)

            for object in ObjectsList:
                if 'person' in object[6]:
                    cv2.putText(r_image, ('{},{}'.format(object[4],object[5])), (object[4]-3,object[5]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # r_image = frame
            # show us frame with detection
            cv2.imshow("Web cam input", r_image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

            # calculate FPS
            fps += 1
            TIME = time.time() - start_time
            if TIME > display_time:
                print("FPS:", fps / TIME)
                fps = 0 
                start_time = time.time()
        except Exception as e:
            logging.error(traceback.format_exc())


    cap.release()
    cv2.destroyAllWindows()
    yolo.close_session()
