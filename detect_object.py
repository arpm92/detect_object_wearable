# import required packages
import cv2
import numpy as np
import os

from model import YOLO


detected_labels = ['person',
                    'bicycle',
                    'car',
                    'motorbike',
                    'bus',
                    'truck',
                    'cat',
                    'dog']

def prepare_yolo_input(img_path,model):
    # read input image
    image = cv2.imread(img_path)

    scale = 0.00392

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # set input blob for the network
    model.model.setInput(blob) # this is the image representation as blob to be the input in the network

    return image, blob


#############

if __name__ == "__main__":

    # region model_parameters

    # define model
    yolo = YOLO()

    # read the classes and asign colors
    classes, COLORS = yolo.classes_colors()

    # endregion

    # region input

    frame_path = 'test_img.png'

    frame, blob =  prepare_yolo_input(frame_path,yolo)

    # endregion

    # region running inference

    image, detected_objects = yolo.model_inference(frame, classes, COLORS, detected_labels, draw_box=True)

    # endregion
    

    # region display results

    # display output image    
    cv2.imshow("object detection", image)

    # wait until any key is pressed
    cv2.waitKey()
        
    # save output image to disk
    cv2.imwrite("object-detection.jpg", image)

    # release resources
    cv2.destroyAllWindows()

    # endregion 