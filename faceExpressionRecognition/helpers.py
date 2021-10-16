import cv2





def face_detection(input):
    cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
    gray_face = None
    input_gray = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    dims = cascade.detectMultiScale(input_gray, 1.15, 6)
    if dims == ():
        print(" no face detected")
    else:
        print(" the image contains a face or more ")
        for (x, y, w, h) in dims:
            cv2.rectangle(input, (x- 20, y- 20), (x + w+20, y + h+20), (0, 0, 255), 3)
            gray_face = input_gray[y-20:y + h+20, x-20:x + w+20]
    return input, gray_face
