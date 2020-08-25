from imutils import face_utils
from ByNumpy_data import *
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2

# 커서의 움직임을 위한 임계값
MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSECUTIVE_FRAMES = 15
EYE_AR_THRESH = 0.23
EYE_AR_CONSECUTIVE_FRAMES = 15
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 4

# 커서 사용 유무에 따른 값 이진화와 프레임 초기화
INPUT_MODE = False
EYE_CLICK = False
LEFT_WINK = False
RIGHT_WINK = False
SCROLL_MODE = False
MOUTH_COUNTER = 0
EYE_COUNTER = 0
WINK_COUNTER = 0
ANCHOR_POINT = (0, 0)

# 이용할 색상 코드 선언
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (250, 237, 125)
RED_COLOR = (241, 95, 95)
GREEN_COLOR = (183, 240, 177)
BLUE_COLOR = (103, 153, 255)
BLACK_COLOR = (0, 0, 0)

# 얼굴을 68개의 점 좌표로 나타낸 Dlib의 파일을 불러와 얼굴 인식의 지표로 삼는다.
shape_predictor = "data/shape_predictor_68_face_landmarks.dat"
# 얼굴을 인식하기 위한 클래스를 만든다(위의 학습된 데이터인 .dat파일을 사용)
detector = dlib.get_frontal_face_detector()
# 인식된 얼굴에서 좌표를 찾기 위한 클래스를 만든다
predictor = dlib.shape_predictor(shape_predictor)

# 웹캠을 가져오고 영상 사이즈를 조정해준다.
vid = cv2.VideoCapture(0)
resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h

# 사용자가 웹캠을 종료할때 까지 반복
while True:
    # 읽어온 영상의 size를 조정하고 grayscale 영상으로 바꿔준다
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # grayscale로 변환된 영상에서 얼굴의 이목구비 인식
    framegray = detector(gray, 0)

    # 찾아낸 부위를 배열에 저장
    if len(framegray) > 0:
        rect = framegray[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    # 얼굴의 영역을 인식하여 영역별로 좌표를 결정하고
    # 각 좌표들을 numpy 형식으로 바꾸고 shape 배열로 만든다
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # 정해진 좌표를 기반으로 배열을 slicing 하여 부위 별로 배열을 설정한다.
    mouth = shape[48:68]
    rightEye = shape[36:42]
    leftEye = shape[42:48]
    nose = shape[27:36]

    # 웹캠의 이미지는 실제 얼굴과 반대가 되기 때문에 데이터를 좌우 반전 시킨다
    temp = leftEye
    leftEye = rightEye
    rightEye = temp

    # 눈과 입의 개패 여부의 평균을 낸다
    mar = mouthOpenClose(mouth)
    leftEAR = eyeOpenClose(leftEye)
    rightEAR = eyeOpenClose(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)

    nose_point = (nose[3, 0], nose[3, 1])

    # convexHull 함수를 이용하여 눈과 입의 좌표들을 연결하여 테두리를 만든다
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
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
    # 만들어진 테두리를 잇는다
    cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

    # 다른 차원으로 저장된 각 좌표들을 2차원 형태로 모두 변환하고
    for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
        # 만들어진 좌표를 circle 함수를 이용하여 표시
        cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)

    # 두 눈의 open 정도의 차이가 thresh 값보다 클경우
    if diff_ear > WINK_AR_DIFF_THRESH:
        # 어떤 눈을 감았는지 판단
        if leftEAR < rightEAR: # 오른쪽 눈의 비율이 크고
            if leftEAR < EYE_AR_THRESH: # open 정도가 thresh 값보다 작을 경우
                WINK_COUNTER += 1 # close 했다고 판단

                if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                    pag.click(button='left') # 마우스 좌클릭

                    WINK_COUNTER = 0 # 초기화

        elif leftEAR > rightEAR: # 왼쪽 눈의 비율이 크고
            if rightEAR < EYE_AR_THRESH: # open 정도가 thresh 값보다 작을 경우
                WINK_COUNTER += 1 # close 했다고 판단

                if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                    pag.click(button='right') # 마우스 우클릭

                    WINK_COUNTER = 0 # 초기화
        else:
            WINK_COUNTER = 0 # 초기화
    else:
        if ear <= EYE_AR_THRESH: # 눈의 open 비율이 thresh 값보다 작을 경우
            EYE_COUNTER += 1 # flag 값 증가

            if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES: # 스크롤 모드 ON
                SCROLL_MODE = not SCROLL_MODE
                EYE_COUNTER = 0 # 초기화

        else:
            EYE_COUNTER = 0 # 초기화
            WINK_COUNTER = 0 # 초기화

    if mar > MOUTH_AR_THRESH: # 입의 개패여부가 thresh 값보다 클 경우 (입을 벌렸을때)
        MOUTH_COUNTER += 1 # flag값 증가

        if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES: # 커서 이동 시작
            INPUT_MODE = not INPUT_MODE
            MOUTH_COUNTER = 0
            ANCHOR_POINT = nose_point # 영상의 중심 = 코의 위치 : 기준점 생성

    else:
        MOUTH_COUNTER = 0 # 초기화

    if INPUT_MODE: # 입을 벌려 input mode가 되었을때
        cv2.putText(frame, "INPUT MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2) # 알림창 생성
        x, y = ANCHOR_POINT # 중심점과
        nx, ny = nose_point # 코의 위치 지정
        w, h = 60, 35 # 스크롤 지표가 될 가로 세로 box
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)# 중심점 기준으로 box draw
        cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)# 영상 중심과 코 위치를 잇는 선

        dir = direction(nose_point, ANCHOR_POINT, w, h) # 좌표의 이동
        print(dir)
        cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2) # 알림창 생성

        drag = 18
        # 위치에 따라 좌우로 마우스 이동
        if dir == 'right':
            pag.moveRel(drag, 0)
        elif dir == 'left':
            pag.moveRel(-drag, 0)

        # 위치에 따라 scroll 방향
        elif dir == 'up':
            if SCROLL_MODE:
                pag.scroll(40)
            else:
                pag.moveRel(0, -drag)
        elif dir == 'down':
            if SCROLL_MODE:
                pag.scroll(-40)
            else:
                pag.moveRel(0, drag)


    if SCROLL_MODE: # 스크롤을 할 경우
        cv2.putText(frame, 'SCROLL MODE ON', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2) # 알림창 생성

    # 화면 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 'ESC' 키를 눌러 프로그램 종료
    if key == 27:
        break

# 프로그램 완전 종료
cv2.destroyAllWindows()
vid.release()