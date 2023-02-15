import cv2

def draw_circular_safe_regions(img,radius_percentage,color):
    """_summary_

    Args:
        img (_type_): image from camera
        radius_p (_type_): percentage of area to cover from max width of the image
        color (_type_): color of the safety area

    Returns:
        _type_: return image
    """
    # safe regions are draw at the center of the lower part of the image

    heigth, width, _ = img.shape

    middle_point = int(width / 2)

    if radius_percentage < 1 and radius_percentage > 0:

        radius = int(width * radius_percentage/2) #pixel value

        img = cv2.circle(img.copy(),(middle_point,int(heigth)),radius,color,10)

    return img

def draw_rectangular_safe_regions(img,zone,color):

    # safe regions are draw at the center of the lower part of the image

    img = cv2.rectangle(img.copy(), (zone[0],zone[1]),(zone[2],zone[3]),color,10)

    return img


def check_object_in_circular_area(img, radius_percentage,object):
    heigth, width, _ = img.shape

    middle_point = int(width / 2)

    if radius_percentage < 1 and radius_percentage > 0:

        radius = int(width * radius_percentage/2) #pixel value

        if object['y'] < heigth and object['y'] > (heigth-radius) and object['x'] > (middle_point-radius) and object['x'] < (middle_point + radius):
            return True # object inside area
    
    return False # Object not in area


def check_object_in_rectangular_area(zone,object):

    zone_width = zone[2] - zone[0]
    zone_height = zone[3] - zone[1]

    if object['x'] > zone[0] and object['x'] < zone[2]:
        if object['y'] > zone[1] and object['y'] < zone[3]:
            if object['w'] > zone_width: # object bigger than safe zone, means closer
                return True # object inside area
    return False  # Object not in area


    