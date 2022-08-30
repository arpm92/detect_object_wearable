import colorsys
import os

from tensorflow.python.platform.tf_logging import warn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time

import traceback
import logging

from yolo_class import YOLO

from safe_region import draw_safe_regions, check_object_in_area


current_path = os.getcwd()


###########

if __name__=="__main__":

    save_path = "YOLOv3-custom-training/output/220819_safe_zones.mp4"
    save_video = False

    # initilize camera
    try:
        video_input = 0
        output_path = os.path.join(current_path,save_path)
        vid = cv2.VideoCapture(video_input) # detect from webcam

        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4
    except Exception as e:
            logging.error(traceback.format_exc())

    #####

    track_only = ["person", "bicycle", "car", "motorbike"]
    yolo = YOLO(track_only=track_only)

    # set start time to current time
    start_time = time.time()
    # displays the frame rate every 2 second
    display_time = 2
    # Set primarry FPS to 0
    fps = 0

    # we create the video capture object cap
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    warning_zone = 0.5
    warning_color = (0,255,255)

    danger_zone = 0.3
    danger_color = (0,0,255)

    while True:
        try:
            ret, frame = cap.read()
            # resize our captured frame if we need
            frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

            # # detect object on our frame
            r_image, ObjectsList = yolo.detect_img(frame)

            r_image = draw_safe_regions(r_image,danger_zone,danger_color)
            
            r_image = draw_safe_regions(r_image,warning_zone,warning_color)

            for object in ObjectsList:
                if 'person' in object[6]:
                    cv2.putText(r_image, ('{},{}'.format(object[4],object[5])), (object[4]-3,object[5]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                if check_object_in_area(r_image,warning_zone,object[4],object[5]):
                    cv2.circle(r_image,(object[4],object[5]),10,warning_color,-1)

                if check_object_in_area(r_image,danger_zone,object[4],object[5]):
                    cv2.circle(r_image,(object[4],object[5]),10,danger_color,-1)

            
            #save vide
            if save_video == True:
                if output_path != '': out.write(r_image)
            

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
