# from sre_constants import SUCCESS
import cv2
import time
import math
import numpy as np
import HandTrackingModule as htm
import mediapipe as mp

# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

# ____________________Below code use for Window os work only___________

# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
# volume.SetMasterVolumeLevel(-20.0, None)
# print(volume.GetVolumeRange())
# volRange = volume.GetVolumeRange()

volRange = (-65, 0.0, 0.03125)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
# ______________Above code use for Window os only work________

# Face Detection Parameter
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imghands = detector.findHands(img)
    hand = detector.handUp(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    faceimg = detector.findFace(img)
    fmList = detector.findFacePositions(img)
    # Hand Mask Operation
    if len(lmList) != 0:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        if 200 < area < 2000:
            # Find Distance b/w index finger and thumb
            length, img, lineinfo = detector.findDistance(4, 8, img)

            # Find value for two point of first finger and thumb
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            # Find Two finger center value
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(imghands, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(imghands, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

            cv2.line(imghands, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # Two finger center circle
            cv2.circle(imghands, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            # Volume Set
            vol = np.interp(length, [50, 300], [minVol, maxVol])
            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])

            # Hand Up
            count_finger = []
            finger = detector.fingersUp()
            if hand[1][0] == "Left":
                # Finger Up
                if finger[4] == 1:
                    count_finger.append(finger[4])
                if finger[3] == 1:
                    count_finger.append(finger[3])
                if finger[2] == 1:
                    count_finger.append(finger[2])
                if finger[1] == 1:
                    count_finger.append(finger[1])
                if finger[0] == 1:
                    count_finger.append(finger[0])
            if hand[1][0] == "Right":
                if finger[4] == 1:
                    count_finger.append(finger[4])
                if finger[3] == 1:
                    count_finger.append(finger[3])
                if finger[2] == 1:
                    count_finger.append(finger[2])
                if finger[1] == 1:
                    count_finger.append(finger[1])
                if finger[0] == 0:
                    count_finger.append(finger[0])

            if len(count_finger) != 0:
                cv2.putText(img, f'{len(count_finger)}', (1000, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            if len(count_finger) == 0:
                cv2.putText(img, f'{len(count_finger)}', (1000, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            # for window only
            # volume.SetMasterVolumeLevel(volPer/100, None)

            if length < 25:
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 15, (255, 255, 255), cv2.FILLED)

    # Face Mask Operation
    if fmList:
        if len(fmList[0]) != 0:

            # find value for two point of first finger and thumb
            x1, y1 = fmList[12][1], fmList[12][2]
            x2, y2 = fmList[14][1], fmList[14][2]

            cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            length = math.hypot(x2 - x1, y2 - y1)
            if length < 8.0:
                # print("Mouth Close")
                cv2.putText(img, f'Mouth Close', (30, 120), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 0), 2)
            if length > 20.0:
                # print("Mouth Open")
                cv2.putText(img, f'Mouth Open', (30, 120), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 0), 2)
    else:
        cv2.putText(img, f'Face Not Detect', (40, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    # Volume Bar Show Program
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'Volume :{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
