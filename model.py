import cv2
import os
import numpy as np


class YOLO():

    def __init__(self,tiny=False) -> None:
        cwdir = os.getcwd()

        if tiny:
            self.conf_path = os.path.join(cwdir,'Yolov3_tiny/yolov3-tiny.cfg')
            self.weight_path = os.path.join(cwdir,'Yolov3_tiny/yolov3-tiny.weights')
            self.classes_path = os.path.join(cwdir,'Yolov3_tiny/coco.names')

        else:
            self.conf_path = os.path.join(cwdir,'YOLOv3-custom-training/model_data/yolov3.cfg')
            self.weight_path = os.path.join(cwdir,'YOLOv3-custom-training/model_data/yolov3.weights')
            self.classes_path = os.path.join(cwdir,'YOLOv3-custom-training/model_data/coco_classes.txt')

        self.model = cv2.dnn.readNet(self.weight_path, self.conf_path)

    # function to get the output layer names 
    # in the architecture
    def get_output_layers(self,net):
        
        layer_names = net.getLayerNames()
        
        # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # working with cv2
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] # working with picamera

        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes, COLORS, detected_labels,w=0,h=0,dist=0):

        label = str(classes[class_id])

        if label in detected_labels:

            color = COLORS[class_id]

            cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

            cv2.putText(img, (str(int(confidence*100))+"%"), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, (label), (x+25,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(img, (f"{w},{h}"), (x-5,y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, (f"distance: {dist}cm"), (x-5,y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def classes_colors(self):
         # read class names from text file
        classes = None
        with open(self.classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # generate different colors for different classes 
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        return classes, COLORS
    
    def calculate_dist(self, w,label):

        dist = 999

        if label == "person":
            # People distance calculation
            #dist = round((0.0156 * w * w - 6.9074 * w + 719.71 + 160),2)
            dist = round((0.0162 * w * w - 7.3712 * w + 792 + 160),2) 

        return dist


    def model_inference(self,image, classes, COLORS, detected_labels, draw_box=True, min_confidence = 0.5):

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
                if confidence > min_confidence:
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

            # i = i[0] # uncomment for cv2
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            obj['x'] = round(x+w/2)
            obj['y'] = round(y+h/2)
            obj['w'] = round(w)
            obj['h'] = round(h)

            obj['ID'] = "unknown"


            label = str(classes[class_ids[i]])

            obj['dist'] = self.calculate_dist(w,label)
 
            if label in detected_labels:

                obj['ID'] = label

            obj_list.append(obj.copy())
            
            if draw_box:
                self.draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes, COLORS, detected_labels,round(w),round(h),obj['dist'])

        return image, obj_list

    def prepare_yolo_input(self,image):

        scale = 1/255.0 #0.00392

        # create input blob 
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), swapRB=True, crop=False)

        # set input blob for the network
        self.model.setInput(blob) # this is the image representation as blob to be the input in the network

        return image, blob
