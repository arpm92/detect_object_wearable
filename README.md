# About

This projects utilizes Yolov3 to detect specific objects on the images. Once the object is detected, it will be tracked and if it goes inside delimited areas, their marker will change on color. The original 80 objects from COCO dataset are in use, but a filter was implemented to display just the items of interest.

The overall idea is to set up a system to recognize specific objects and their location and stablish safe zones, where if certain objects get-in, it will be used as a trigger for further actions.

### Project still on development.

# SetUp

## Download YOLOv3 weights from the YOLO website, or use wget command:

```wget https://pjreddie.com/media/files/yolov3.weights```

https://pjreddie.com/darknet/yolo/

## Copy downloaded weights file to model_data folder and convert the Darknet YOLO model to a Keras model:

```python convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo_weights.h5```

## Run Program

Run webcam_detect.py 
