import cv2


class Picture:
    def __init__(self, name):
        self.name = name

    def takepic(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            cv2.imshow("prees enter to take image ", frame)
            if cv2.waitKey(1) == 13:
                cv2.imwrite(self.name + ".jpg", frame)
                break
        cap.release()
        cv2.destroyAllWindows()
