import cv2

def draw_safe_regions(img,radius_percentage,color):
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

        img = cv2.circle(img,(middle_point,int(heigth)),radius,color,10)

    return img


def check_object_in_area(img, radius_percentage,object_x, object_y):
    heigth, width, _ = img.shape

    middle_point = int(width / 2)

    in_x = None
    in_y = None

    if radius_percentage < 1 and radius_percentage > 0:

        radius = int(width * radius_percentage/2) #pixel value

        if object_y < heigth and object_y > (heigth-radius) and object_x > (middle_point-radius) and object_x < (middle_point + radius):
            return True # object inside area
    
    return False # Object not in area

    