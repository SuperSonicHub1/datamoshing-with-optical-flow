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

# https://easings.net/#easeInSine
def ease_in_sine(x: float) -> float:
	return 1 - np.cos(x * np.pi / 2)

def well(frames: int, image: np.ndarray, writer: cv.VideoWriter, fps: int = 24, m2: float = 1e16):
	G = 6.674e-11

	dt =  1. / fps
	t = .0
	
	height, width, _ = np.shape(image)

	datamoshed = image
	
	for i in range(frames):
		print(i)

		mapping = np.zeros((height, width, 2), dtype=np.float32)
		for y in range(height):
			for x in range(width):
				mapping[y, x, 0] = x
				mapping[y, x, 1] = y

		displacement = np.zeros((height, width, 2), dtype=np.float32)
		for x in range(height):
			for y in range(width):
				progress = i / frames
				current_m2 = ease_in_sine(progress) * m2

				r = np.array([x - height/2, width/2 - y])
				r_len = np.linalg.norm(x)
				# v(0) = 0
				v = -G * current_m2 / np.max([r_len * r_len * r_len, 0.000001]) * r * t
				v_dt = v * dt
				displacement[x, y, :] = v

		# TODO: why the negation?
		mapping -= displacement

		datamoshed = remap(datamoshed, mapping)
		writer.write(datamoshed)

		t += dt
