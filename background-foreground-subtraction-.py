import cv2
import numpy as np

cap = cv2.VideoCapture(0);

#to initiliaze running average
ret, frame_ = cap.read()

#to get background only
average = np.float32(frame_)

foreground_background = cv2.createBackgroundSubtractorMOG2()

while True:
	ret, frame = cap.read()

	foreground_mask = foreground_background.apply(frame)

	cv2.accumulateWeighted(frame, average, 0.01)
	background_mask = cv2.convertScaleAbs(average)

	cv2.imshow("Input", frame)
	cv2.imshow("Foreground", foreground_mask)
	cv2.imshow("Background", background_mask)
	if cv2.waitKey(1) == 13:
		break;

cap.release()
cv2.destroyAllWindows()
