import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_img(filename):
	image_template = cv2.imread(filename)
	gray_template = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)

	row, col = 1, 2
	fig, axs = plt.subplots(row, col, figsize=(15,10))
	fig.tight_layout()

	axs[0].imshow(cv2.cvtColor(image_template, cv2.COLOR_BGR2RGB))
	axs[0].set_title("Where is Waldo?")

	plt.show()

	return image_template, gray_template

def load_waldo(filename, image_template):
	gray_waldo = cv2.imread(filename, 0)
	color_waldo = cv2.imread(filename)

	row, col = 1, 2
	fig, axs = plt.subplots(row, col, figsize=(5, 3))
	fig.tight_layout()

	axs[0].imshow(cv2.cvtColor(image_template, cv2.COLOR_BGR2RGB))
	axs[0].set_title("Waldo.jpg")
	cv2.imwrite('waldo.jpg', color_waldo)

	plt.show()
	return color_waldo, gray_waldo

def find_waldo(gray_template, gray_waldo):
	result = cv2.matchTemplate(gray_template, gray_waldo, cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

	#make bounding box

	top_left = max_loc
	bottom_right = (top_left[0] + 50, top_left[1] + 50)
	cv2.rectangle(image_template, top_left, bottom_right, (0,0,255), 5)

	plt.imshow(cv2.cvtColor(image_template, cv2.COLOR_BGR2RGB))
	plt.title("Where is Waldo?")
	plt.show()

image_template, gray_template = load_img('WaldoBeach.jpeg')
_, gray_waldo = load_waldo('waldo.jpeg', image_template)
find_waldo(gray_template, gray_waldo)
