import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import pytesseract as tess
#Testing tess
'''
img = cv2.imread('/home/kmd/openCV-tutorials/images/tess_test.png')

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Test img');
plt.show()

test_image = Image.fromarray(img)
text = tess.image_to_string(test_image, lang = 'eng')
print("Text detected: " + text)
'''

def preprocessing(img):
	blur_img = cv2.GaussianBlur(img, (5,5), 0)
	gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

	sobelx = cv2.Sobel(gray, cv2.CV_8U,1,0,ksize = 3)
	_, threshold_img = cv2.threshold(sobelx, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	return threshold_img

def get_contours(threshold_img):
	element = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (17,3))
	morph_img_threshold = threshold_img.copy()
	cv2.morphologyEx(src = threshold_img, op = cv2.MORPH_CLOSE, kernel = element, dst = morph_img_threshold)
	contours, _ = cv2.findContours(morph_img_threshold, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

	return contours

def check_ratio(area, width, height):
	ratio = float(width)/float(height)
	if ratio < 1:
		ratio = 1 /ratio
	aspect = 4.7272
	min_ar = 15*aspect*15
	max_ar = 125*aspect*125

	rmin = 3
	rmax = 6

	if (area < min_ar or area > max_ar) or (ratio < rmin or ratio > rmax):
		return False
	return True

def get_plate_contours(plate):
	gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
	thresh = cv2.dilate(gray, kernel, iterations = 1)

	_, thresh = cv2.threshold(gray, 150,225, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	if contours:
		areas = [cv2.contourArea(c) for c in contours]
		max_indx = np.argmax(areas)

		max_contour = contours[max_indx]
		max_contour_area = areas[max_indx]
		x,y,w,h = cv2.boundingRect(max_contour)

		if not check_ratio(max_contour_area, w, h):
			return plate, None

		cleaned_final = thresh[y:y+h, x:x+w]
		plt.imshow(cv2.cvtColor(cleaned_final, cv2.COLOR_BGR2RGB))
		plt.title('Function Test');
		plt.show()

		return cleaned_final, [x,y,w,h]
	else:
		return plate, None

def check_rect_angle(rect):
	(x,y), (w,h), rect_angle = rect
	if(w > h):
		angle = -rect_angle
	else:
		angle = 90 + rect_angle
	if angle > 15:
		return False
	if h == 0 or w == 0:
		return False
	area = h*w
	if not check_ratio(area, w, h):
		return False
	else:
		return True

def is_max_white(plate):
	avg = np.mean(plate)
	if avg >= 115:
		return True
	else:
		return False

def clean_and_read(img, contours):
	#OCR 
	for i, contour in enumerate(contours):
		min_rect = cv2.minAreaRect(contour)
		if check_rect_angle(min_rect):
			x,y,w,h = cv2.boundingRect(contour)
			plate_img = img[y:y+h, x:x+w]

			if(is_max_white(plate_img)):
				clean_plate, rect = get_plate_contours(plate_img)

				if rect:
					row, col = 1, 2
					x1,y1, w1, h1 = rect
					x,y, w, h = x+x1, y+y1, w1, h1

					plate_img = Image.fromarray(plate_img)
					text = tess.image_to_string(plate_img, lang='eng')
					print('Detected Text:' + text)

if __name__=='__main__':
	print("DETECTING PLATE....")
	img = cv2.imread('/home/kmd/openCV-tutorials/images/car2.jpeg')
	print(img)
	threshold_image = preprocessing(img)
	contours = get_contours(threshold_image)
	clean_and_read(img, contours)
	cv2.destroyAllWindows()
