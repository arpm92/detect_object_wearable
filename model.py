import cv2
import os
import numpy as np


class YOLO():

    def __init__(self) -> None:
        cwdir = os.getcwd()

        self.conf_path = os.path.join(cwdir,'YOLOv3-custom-training/model_data/yolov3.cfg')
        self.weight_path = os.path.join(cwdir,'YOLOv3-custom-training/model_data/yolov3.weights')
        self.classes_path = os.path.join(cwdir,'YOLOv3-custom-training/model_data/coco_classes.txt')

        self.model = cv2.dnn.readNet(self.weight_path, self.conf_path)

    # function to get the output layer names 
    # in the architecture
    def get_output_layers(self,net):
        
        layer_names = net.getLayerNames()
        
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes, COLORS, detected_labels):

        label = str(classes[class_id])

        if label in detected_labels:

            color = COLORS[class_id]

            cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

            cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def classes_colors(self):
         # read class names from text file
        classes = None
        with open(self.classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # generate different colors for different classes 
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        return classes, COLORS

    def model_inference(self,image, classes, COLORS, detected_labels, draw_box=True):

        obj = {}

        obj_list = []

        Width = image.shape[1]
        Height = image.shape[0]

        # run inference through the network
        # and gather predictions from output layers
        outs = self.model.forward(self.get_output_layers(self.model))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])



        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            obj['x'] = round(x+w/2)
            obj['y'] = round(y+h/2)
            obj['w'] = round(w)
            obj['h'] = round(h)

            label = str(classes[class_ids[i]])

            if label in detected_labels:

                obj['ID'] = label

            obj_list.append(obj.copy())
            
            if draw_box:
                self.draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes, COLORS, detected_labels)

        return image, obj_list

    def prepare_yolo_input(self,image):

        scale = 0.00392

        # create input blob 
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        # set input blob for the network
        self.model.setInput(blob) # this is the image representation as blob to be the input in the network

        return image, blob