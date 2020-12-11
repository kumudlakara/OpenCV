import cv2
import numpy as np
import imutils
from imutils import contours

reference = cv2.imread('#ocr-ref_file_name')
reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
reference = cv2.threshold(reference, 110, 255, cv2.THRESH_BINARY_INV)[1]

ref_contours = cv2.findContours(reference.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ref_contours = imutils.grab_contours(ref_contours)
ref_contours = contours.sort_contours(ref_contours, method="left-to-right")[0]
digits = {}

for (i,c) in enumerate(ref_contours):
	(x,y,w,h) = cv2.boundingRect(c)
	roi = reference[y:y+h, x:x+w]
	roi = cv2.resize(roi, (57,88))

	digits[i] = roi

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

image = cv2.imread('#credit_card_01.png')
image = imutils.resize(image, width = 300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

w_on_b = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)

gradx = cv2.Sobel(w_on_b, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradx = np.absolute(gradx)
(minv, maxv) = (np.min(gradx), np.max(gradx))
gradx = (255*((gradx - minv)/(maxv - minv)))
gradx = gradx.astype("uint8")

gradx = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, rect_kernel)
thresh = cv2.threshold(gradx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)

cntrs = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntrs = imutils.grab_contours(cntrs)
locations = []

for (i,c) in enumerate(cntrs):
	(x,y,w,h) = cv2.boundingRect(c)
	aspect_ratio = w/float(h)

	if aspect_ratio > 2.5 and aspect_ratio < 4:
		if (w>40 and w<55) and (h>10 and h<20):
			locations.append((x,y,w,h))

locations = sorted(locations, key= lambda x:x[0])
output = []

for (i,(gx,gy,gw,gh)) in enumerate(locations):
	groupop = []

	group = gray[gy - 5: gy + gh + 5, gx -5: gx+gw+5]
	group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	dig_contours = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	dig_contours = imutils.grab_contours(dig_contours)
	dig_contours = contours.sort_contours(dig_contours, method="left-to-right")[0]

	for c in dig_contours:
		(x,y,w,h) = cv2.boundingRect(c)
		roi = group[y:y+h, x:x+w]
		roi = cv2.resize(roi, (57,88))

		scores = []

		for (digit, digitroi) in digits.items():
			result = cv2.matchTemplate(roi, digitroi, cv2.TM_CCOEFF)
			_, score, _, _ = cv2.minMaxLoc(result)
			scores.append(score)

		groupop.append(str(np.argmax(scores)))

	cv2.rectangle(image, (gx-5, gy-5), (gx+gw+5, gy+gh+5),(0,0,255),2)
	cv2.putText(image, "".join(groupop), (gx, gy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

	output.extend(groupop)

	FIRST_NUMBER = {
	"3":"American Express",
	"4":"Visa",
	"5":"MasterCard"
	}

	print("Credit Card type: {}".format(FIRST_NUMBER[output[0]]))
	print("Credit Card number: {}".format("".join(output)))
	
cv2.imshow("CardImage", image)

if cv2.waitKey() == 13:
	cv2.destroyAllWindows()

