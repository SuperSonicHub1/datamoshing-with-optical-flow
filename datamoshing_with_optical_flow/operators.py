import cv2 as cv
import numpy as np

def flow(previous, next):
	"""
	We first compute optical flow for 𝑉 in order to define the perceived
	motion between each frame. We denote this as an operation be-
	tween two frames called 𝑓 𝑙𝑜𝑤.

	The authors look to be referring to a dense optical flow:
	the flow of all points in the frame. Here, we use OpenCV's
	implementation of Gunnar Farnebäck's estimator for a dense
	optical flow.

	https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
	https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
	https://link.springer.com/chapter/10.1007/3-540-45103-X_50
	"""

	previous_grey = cv.cvtColor(previous, cv.COLOR_BGR2GRAY)
	next_grey = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
	displacement = cv.calcOpticalFlowFarneback(
		previous_grey,
		next_grey,
		None,
		# TODO: Play with paramaters; see what works!
		0.5,
		3,
		15,
		3,
		5,
		1.2,
		0
	)


	width, height, _ = np.shape(previous)
	flow_mapping = np.zeros_like(displacement)
	for x in range(width):
		for y in range(height):
			# TODO: why the swap?
			flow_mapping[x, y, 0] = y
			flow_mapping[x, y, 1] = x
	# TODO: why the negation?
	flow_mapping -= displacement
	
	return flow_mapping


def remap(image: np.ndarray, mapping: np.ndarray):
	return cv.remap(image, mapping[..., 0], mapping[..., 1], cv.INTER_NEAREST)
