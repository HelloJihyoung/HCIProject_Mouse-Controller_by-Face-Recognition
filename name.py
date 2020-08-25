import imutils
from utils import *
import numpy as np
import pyautogui as pag
import cv2

# Thresholds and consecutive frame length for triggering the mouse action.
MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSECUTIVE_FRAMES = 15
EYE_AR_THRESH = 0.19
EYE_AR_CONSECUTIVE_FRAMES = 15
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 10

# Initialize the frame counters for each action as well as
# booleans used to indicate if action is performed or not
MOUTH_COUNTER = 0
EYE_COUNTER = 0
WINK_COUNTER = 0
INPUT_MODE = False
EYE_CLICK = False
LEFT_WINK = False
RIGHT_WINK = False
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)

# Video capture
vid = cv2.VideoCapture(0)
img = cv2.imread('1.png')
mask = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret ,thr1 = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thr1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
mmt = cv2.moments(cnt)
for key, value in mmt.items():
    #print(key," : ",value)
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])
    nose_point = (cx, cy)
    #print( 'x 무게중심', cx, 'y 무게중심', cy )

resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h
#cv2.imshow("Frame", mask)




while True:
    # Grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    maskFrame = cv2.add(mask, frame)
    gray = cv2.cvtColor(maskFrame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 15)
    blur = cv2.GaussianBlur(blur, (3, 3), 0)
    ret, thresh = cv2.threshold(blur, 40, 255, 0)
    center, old_center = [[0, 0]] * 2, [[0, 0]] * 2
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(maskFrame, contours, -1, YELLOW_COLOR, 1)

    cv2.imshow("Frame", maskFrame)
    key = cv2.waitKey(1) & 0xFF

    # 'ESC' 키를 눌러 프로그램 종료
    if key == 27:
         break

# 프로그램 완전 종료
cv2.destroyAllWindows()
vid.release()