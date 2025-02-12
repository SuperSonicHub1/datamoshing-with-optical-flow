import cv2 as cv
from datamoshing_with_optical_flow import datamosh

# video = cv.VideoCapture("A Self driving truck on a Chinese highway [g8pyetqmqece1].mp4")
# video = cv.VideoCapture("DSCF6216.mp4")
video = cv.VideoCapture("Explosion croma key green screen [RMvMYYRDpKM].webm")
frame_rate = video.get(cv.CAP_PROP_FPS)
width = video.get(cv.CAP_PROP_FRAME_WIDTH)
height = video.get(cv.CAP_PROP_FRAME_HEIGHT)

# image = cv.imread("test_image.png")
# image = cv.imread("test_image_2.png")
image = cv.imread("test_image_3.png")

# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
fourcc = cv.VideoWriter_fourcc(*'XVID') # type: ignore
writer = cv.VideoWriter('output.avi', fourcc, frame_rate, (int(width), int(height)))
datamosh(video, image, writer)

# TODO: Use contextlib
video.release()
writer.release()

