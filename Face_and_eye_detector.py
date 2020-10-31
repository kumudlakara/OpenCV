import numpy as np
import cv2
from matplotlib import pyplot as plt

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifer = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_face(img, size = 0.5):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.1, 3)
	if faces is ():
		return img

	for (x,y,w,h) in faces:
		x -= int(size*10)
		w += int(size*10)
		y -= int(size*10)
		h += int(size*10)
		cv2.rectangle(img, (int(x),int(y)), (int(x+w), int(y+h)), (0, 255, 0), 5)
		roi_gray = gray
		roi_img = img
		eyes = eye_classifer.detectMultiScale(roi_gray, 1.2, 6)

		#draw rectangle for eyes aswell
		for (ey_x, ey_y, ey_w, ey_h) in eyes:
			cv2.rectangle(roi_img, (ey_x, ey_y), (ey_x + ey_w, ey_y + ey_h), (255, 0, 0), 5)
		roi_img = cv2.flip(roi_img,1)
		return roi_img

cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	if not ret:
		print("Unable to receive video input")
		break

	cv2.imshow("FACE DETECTOR", detect_face(frame))

	if cv2.waitKey(1) == ord('q'):     
		break

cv2.release()
cv2.destroyAllWindows()


