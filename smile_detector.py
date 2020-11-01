import cv2
import dlib
import numpy as np
from imutils import face_utils
import math

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


coords = np.zeros((68,2), dtype = int)
cap = cv2.VideoCapture(0)

def lip_right_corner(landmarks):
	top_lip_pts = []
	for i in range(49,51,53):
		top_lip_pts.append(landmarks[i])
	for i in range(60, 62):
		top_lip_pts.append(landmarks[i])
	top_lip_mean = np.mean(top_lip_pts, axis = 0)
	return top_lip_mean

def lip_left_corner(landmarks):
	bottom_lip_pts = []
	for i in range(54,57):
		bottom_lip_pts.append(landmarks[i])
	for i in range(65,66):
		bottom_lip_pts.append(landmarks[i])
	bottom_lip_mean = np.mean(bottom_lip_pts, axis = 0)
	return bottom_lip_mean 


def lip_corner_dist(pt1, pt2):
	return (math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2))

while True:
	ret, frame = cap.read()
	if not ret:
		print("Unable to receive frame stream. Aborting")

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray)


	for face in faces:
		x1 = face.left()
		y1 = face.top()
		x2 = face.right()
		y2 = face.bottom()

		landmarks = predictor(gray, face)

		for i in range(0, 68):
			coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

	lipR_mean = lip_right_corner(coords)
	lipL_mean = lip_left_corner(coords)
	if(lip_corner_dist(lipR_mean, lipL_mean) > 40):
		cv2.putText(frame, "Subject is Smiling :)", (50,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)


	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) == 13:
		break






