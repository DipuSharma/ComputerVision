import cv2
import mediapipe as mp
import time
import math

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    xList = []
    yList = []
    lmlist = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
        for id, lm in enumerate(faceLms.landmark):
            # print(lm)
            ih, iw, ic = img.shape
            x, y = int(lm.x * iw), int(lm.y * ih)
            xList.append(x)
            yList.append(y)
            lmlist.append([id, x, y])

    if len(lmlist[0]) != 0:

        # find value for two point of first finger and thumb
        x1, y1 = lmlist[12][1], lmlist[12][2]
        x2, y2 = lmlist[14][1], lmlist[14][2]

        cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        if length < 4.0:
            # print("Mouth Close")
            cv2.putText(img, f'Mouth Close', (40, 150), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 0, 0), 3)
        if length > 20.0:
            # print("Mouth Open")
            cv2.putText(img, f'Mouth Open', (40, 150), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
