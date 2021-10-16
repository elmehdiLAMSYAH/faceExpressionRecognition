from face_expression import FacialExpressionModel
from take_pic import Picture
from helpers import face_detection
import cv2


model_path = 'model.json'
weights_path = 'model_weights.h5'
model = FacialExpressionModel(model_path, weights_path)

picture = Picture('me')
picture.takepic()
image = cv2.imread("me.jpg")
color, gray = face_detection(image)
print(color.shape)
cv2.imshow("face color ", color)
cv2.waitKey(0)
print(gray.shape)
cv2.imshow(" gray face ", gray)
cv2.waitKey(0)
gray_face = cv2.resize(gray, (48, 48))
prediction = model.predict_emotion(gray_face)
cv2.putText(color, prediction, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
cv2.imshow("face color ", color)
cv2.waitKey(0)
cv2.destroyAllWindows()