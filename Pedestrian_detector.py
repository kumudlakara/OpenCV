import cv2
import numpy as np

person_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
	print("Failed to open camera")

while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR)

	if not ret:
		print("Failed to receive frame stream. Aborting");
		break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	pedestrians = person_detector.detectMultiScale(frame, 1.2, 3)

	for (x,y,w,h) in pedestrians:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 5)
		cv2.imshow("Pedestrian Detector", frame);

	if cv2.waitKey(1) == 13:
		break;

cv2.release()
cv2.deleteAllWindows()
