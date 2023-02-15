import cv2


def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


def cam(input=0):
    cap = cv2.VideoCapture(input)
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
        cv2.imshow('webcam', frame)# press escape to exit
        if (cv2.waitKey(30) == 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    cam_index = returnCameraIndexes()

    print(cam_index)


    cam(0)