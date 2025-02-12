import cv2 as cv
import numpy as np

from .operators import remap, flow


def datamosh(video: cv.VideoCapture, image: np.ndarray, writer: cv.VideoWriter):
	ret, previous = video.read()
	if not ret:
		return
	ret, next = video.read()
	if not ret:
		return

	width, height, _ = np.shape(previous)

	flow_map = flow(previous, next)

	remapped = remap(image, flow_map)
	mask = remap(np.ones_like(remapped), flow_map)
	datamoshed = (mask * remapped) + (1 - mask) * next
	writer.write(datamoshed)

	while True:
		previous = next
		ret, next = video.read()
		if not ret:
			return

		flow_map = flow(previous, next)
		remapped = remap(datamoshed, flow_map)
		mask = remap(mask, flow_map)
		datamoshed = (mask * remapped) + (1 - mask) * next
		writer.write(datamoshed)
