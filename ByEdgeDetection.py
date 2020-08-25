import numpy as np
import pyautogui as pag
import imutils
import cv2
import math
EYE_AR_THRESH = 0.23
EYE_AR_CONSECUTIVE_FRAMES = 15
WINK_AR_DIFF_THRESH = 50
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 3

# 커서의 움직임을 위한 임계값
MOUTH_COUNTER = 0
EYE_COUNTER = 0
WINK_COUNTER = 0
INPUT_MODE = False
EYE_CLICK = False
LEFT_WINK = False
RIGHT_WINK = False
SCROLL_MODE = False
ANCHOR_POINT = (270, 260)

# 이용할 색상 코드 선언
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (250, 237, 125)
RED_COLOR = (241, 95, 95)
GREEN_COLOR = (183, 240, 177)
BLUE_COLOR = (103, 153, 255)
BLACK_COLOR = (0, 0, 0)

# 웹캠을 가져오고 영상 사이즈를 조정해준다.
vid = cv2.VideoCapture(0)
cam_w = 640
cam_h = 480
resolution_w = 1366
resolution_h = 768
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h

# 얼굴 인식을 위한 마스크 호출 및 사이즈 조정
img = cv2.imread('1.png')
mask = cv2.resize(img, dsize=(cam_w, cam_h), interpolation=cv2.INTER_AREA)

# 마스크를 생성하여 처음 만들어지는 컨투어 (오른쪽 눈)의 무게 중심을 기준점으로 잡는다
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret ,thr1 = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thr1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
mmt = cv2.moments(cnt)

# 무게중심을 구하여 con_point_old에 좌표를 저장한다
for key, value in mmt.items():
    #print(key," : ",value)
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])
    con_point_old = (cx, cy)
    #print( 'x 무게중심', cx, 'y 무게중심', cy )


# 사용자가 웹캠을 종료할때 까지 반복
while True:
    # 읽어온 영상의 size를 조정하고 grayscale 영상으로 바꿔준다
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    # 위에서 생성한 마스크와 불러온 웹캠의 frame을 합침
    maskFrame = cv2.add(mask, frame)
    # edge 검출을 위해 grayscale 이미지로 변환
    gray = cv2.cvtColor(maskFrame, cv2.COLOR_BGR2GRAY)
    # 이목구비 외에 얼굴의 다른 점과 같은 노이즈 제거를 위해 blur 처리
    blur = cv2.medianBlur(gray, 15)
    blur = cv2.GaussianBlur(blur, (3, 3), 0)


    # thresholding과 canny, sobel edge detection을 사용하여 이목구비 검출
    # ret, thresh = cv2.threshold(blur, 40, 255, 0)
    #img_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    #img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    #img_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    #img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
    #img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0);
    # canny가 가장 효율적이였음
    canny = cv2.Canny(blur, 100, 200)
    # 기준점을 .line 함수를 이용하여 draw
    cv2.line(maskFrame, (270, 0), (270, 640), GREEN_COLOR, 3)
    cv2.line(maskFrame, (0, 260), (480, 260), GREEN_COLOR, 3) #(270,260)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 새로운 중심 좌표 바뀔때마다 검출
    for key, value in mmt.items():
        # print(key," : ",value)
        cx = int(mmt['m10'] / mmt['m00'])
        cy = int(mmt['m01'] / mmt['m00'])
        con_point_new = (cx, cy)

    # 각 컨투어에 인덱스 부여
    rightEye = contours[1]
    leftEye = contours[0]
    mouth = contours[2]

    # 컨투어한 영역의 넓이값 계산
    leftarea = cv2.contourArea(leftEye)
    rightarea = cv2.contourArea(rightEye)
    moutharea = cv2.contourArea(mouth)
    avgarea = (leftarea + rightarea) / 2
    diff_ear = np.abs(leftarea - rightarea)

    # convexHull 함수를 통해 테두리 생성
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    mouthHull = cv2.convexHull(mouth)
    """
       # 테두리를 만드는  다른 함수인 approxPolyDP를 이용하여 점으로 된 부분을 곡선의 형태로 테두리 생성
       epsilon = 0.03
       epsilon_mouth = epsilon * cv2.arcLength(mouth, True)
       mouthHull = cv2.approxPolyDP(mouth, epsilon_mouth, True)
       epsilon_left = epsilon * cv2.arcLength(leftEye, True)
       leftEyeHull = cv2.approxPolyDP(leftEye, epsilon_left, True)
       epsilon_right = epsilon * cv2.arcLength(rightEye, True)
       rightEyeHull = cv2.approxPolyDP(rightEye, epsilon_left, True)
     """
    # 위에서 생성한 테두리 그리기
    cv2.drawContours(maskFrame, [leftEyeHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(maskFrame, [rightEyeHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(maskFrame, [mouthHull], -1, YELLOW_COLOR, 1)

    if diff_ear > 200:  # 둘의 면적 차이가 200 이상일 경우 한쪽 눈을 감았다고 가정

        if leftarea < rightarea:  # 왼쪽 눈의 면적이 오른쪽의 면적보다 작으며
            if leftarea < 200:  # 왼쪽 눈의 면적이 200보다 작으면
                pag.click(button='left')  # 왼쪽눈을 감아 좌클릭을 했다고 판단

        elif leftarea > rightarea:  # 오른쪽 눈의 면적이 왼쪽 눈의 면적보다 작으며

            if rightarea < 200:  # 오른쪽 눈의 면적이 200 보다 작을 경우
                pag.click(button='right')  # 오른쪽 눈을 감아 우클릭을 했다고 판단
    else:
        if avgarea <= 150:  # 양쪽 눈의 면적 크기의 영역 평균이 150 보다 작을 경우
            # 두 눈을 모두 감았다고 판단하여 스크롤 모드 ON
            SCROLL_MODE = not SCROLL_MODE

        else:
            EYE_COUNTER = 0  # 초기화
            WINK_COUNTER = 0  # 초기화

    if len(contours) >= 6:  # contour의 개수가 6 이상일때 입을 벌린 것으로 판단하여
        INPUT_MODE = not INPUT_MODE  # 커서 이동 시작
        ANCHOR_POINT = con_point_old  # 영상의 중심 = 영상 눈의 위치 : 기준점 생성

    else:
        MOUTH_COUNTER = 0  # 초기화

    if INPUT_MODE:  # 입을 벌려 input mode가 되었을때
        cv2.putText(frame, "INPUT MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)  # 알림창 생성
        x, y = ANCHOR_POINT  # 얼굴 중앙 점
        ox, oy = con_point_old
        nx, ny = con_point_new  # 기준 위치 지정(눈좌표)

        w, h = 60, 35  # 스크롤 지표가 될 가로 세로 box
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)  # 중심점 기준으로 box draw
        cv2.line(frame, ANCHOR_POINT, con_point_old, BLUE_COLOR, 2)  # 영상 중심과 코 위치를 잇는 선
        drag = 18

        # 새로운 중심의 좌표와 이전의 중심 좌표를 비교해가며 마우스 커서 이동과 스크롤 관리
        if nx > ox:
            pag.moveRel(drag, 0)
        elif nx < ox:
            pag.moveRel(-drag, 0)

        # 위치에 따라 scroll 방향
        elif ny < oy:
            if SCROLL_MODE:
                pag.scroll(40)
            else:
                pag.moveRel(0, -drag)
        elif ny > oy:
            if SCROLL_MODE:
                pag.scroll(-40)
            else:
                pag.moveRel(0, drag)


    if SCROLL_MODE:  # 스크롤을 할 경우
        cv2.putText(frame, 'SCROLL MODE ON', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)  # 알림창 생성

    # 화면 출력
    cv2.imshow("Frame", maskFrame)
    key = cv2.waitKey(1) & 0xFF

    # 'ESC' 키를 눌러 프로그램 종료
    if key == 27:
         break

# 프로그램 완전 종료
cv2.destroyAllWindows()
vid.release()