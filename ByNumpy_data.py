import numpy as np

# 눈의 비율 계산을 통해 마우스 클릭을 담당
def eyeOpenClose(eye):
    # 좌표를 이용하여 두 점 사이의 거리를 구한다
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # 눈의 좌표를 이용하여 같은 가로선 상에 있는 거리를 구한다
    C = np.linalg.norm(eye[0] - eye[3])

    # 눈의 비율을 계산
    ear = (A + B) / (2.0 * C)

    # 눈의 비율을 계산하여 눈이 떠졌는지 감겼는지 판단
    return ear

def mouthOpenClose(mouth):
    # 좌표를 이용하여 두 점 사이의 거리를 구한다
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])

    # 입술의 좌표를 이용하여 같은 가로선 상에 있는 거리를 구한다
    D = np.linalg.norm(mouth[12] - mouth[16])

    # 입의 비율을 계산하여 입이 열렸는지 닫혔는지 판단
    mar = (A + B + C) / (2 * D)

    # Return the mouth aspect ratio
    return mar

# 코의 위치와 고정점의 위치의 거리를 계산하여 마우스 커서 이동을 판단
def direction(nose_point, anchor_point, w, h, multiple=1):
    nx, ny = nose_point
    x, y = anchor_point

    if nx > x + multiple * w:
        return 'right'
    elif nx < x - multiple * w:
        return 'left'

    if ny > y + multiple * h:
        return 'down'
    elif ny < y - multiple * h:
        return 'up'

    return '-'
