import cv2
from matplotlib.pyplot import draw
import mediapipe as mp
import time
import numpy as np
import math
import pygame
from pygame import mixer

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
sTime = time.time()

finger_pos = []
can_hit = 4

recording = []

dpx = 0
dpy = 0

pygame.init()

pygame.mixer.set_num_channels(2)

snare = mixer.Sound('snare.wav')
kick = mixer.Sound('kick.wav')


def draw_hands(img, results, fps):
    global finger_pos
    global can_hit
    global drum_pos
    hit = -1
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):

                if(id == 8):
                    finger_pos.append(math.sqrt(lm.x**2+lm.y**2))
                    if len(finger_pos) > 4:
                        finger_pos.pop(0)
                    diff_finger_pos = np.diff(np.diff(finger_pos)/fps)/fps
                    h, w, c = img.shape
                    print(diff_finger_pos)
                    if can_hit < 1 and abs(lm.x * w - dpx) < .05 * w and abs(lm.y * h - dpy) < 0.05 * h and diff_finger_pos[0] < -0.00011:
                        can_hit = 2
                        hit = int(lm.x * w), int(lm.y*h)
                        print(id, lm)
                    else:
                        can_hit -= 1

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            return hit


def draw_fps(img):
    global cTime
    global pTime
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return fps


while(True):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    fps = draw_fps(img)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                if(id == 8):
                    h, w, c = img.shape
                    dpx = int(lm.x * w)
                    dpy = int(lm.y * h)
    cv2.imshow('test', img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

sTime = time.time()

while(True):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    cv2.circle(img, (dpx, dpy), 15, (0, 255, 0), -1)

    fps = draw_fps(img)
    hit = draw_hands(img, results, fps)

    if hit != -1 and hit != None:
        snare.play()
        cv2.circle(img, hit, 20, (255, 0, 0), -1)
        recording.append((time.time()-sTime, 1))
        print(recording)

    cv2.imshow('test', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

sTime = time.time()
i = 0
while(True):
    if abs((time.time()-sTime)-recording[i][0]) < .005:
        print("BANG!!")
        i += 1
        if(i == len(recording)):
            break
