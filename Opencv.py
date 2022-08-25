# from sre_constants import SUCCESS
import cv2
import time
import math
import numpy as np
import HandTrackingModule as htm
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

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList[0]) != 0:
        # print(lmList[0][4], lmList[0][8])

        # find value for two point of first finger and thumb
        x1, y1 = lmList[0][4][1], lmList[0][4][2]
        x2, y2 = lmList[0][8][1], lmList[0][8][2]
        # Find Two finger center value
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        # Two finger center circle
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        # for window only
        # volume.SetMasterVolumeLevel(vol, None)

        if length < 25:
            cv2.circle(img, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'Volume :{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
