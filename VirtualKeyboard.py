import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import cvzone
import numpy as np
from pynput.keyboard import Controller

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

finalText = ""
keyboard = Controller()

# def drawAll(img, buttonList):
#     for button in buttonList:
#         x, y = button.pos
#         w, h = button.size
#         cv.rectangle(img, button.pos, (x+w, y + h), (255, 0, 255), cv.FILLED)
#         cv.putText(img, button.text, (x + 25, y + 65), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
#     return img

def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0)
        cv.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]),
                      (255, 0, 255), cv.FILLED)
        cv.putText(imgNew, button.text, (x + 40, y + 60),
                    cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    print(mask.shape)
    out[mask] = cv.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

class Button():
    def __init__(self, pos, text, size = [85, 85]):
        self.pos = pos
        self.text = text
        self.size = size

buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    hands, img = detector.findHands(img)

    img = drawAll(img, buttonList)

    if hands and len(hands[0]['lmList']) >= 9:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1['center']
        handType1 = hand1["type"]
        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"] 
            centerPoint2 = hand2['center']
            handType2 = hand2["type"]

            fingers2 = detector.fingersUp(hand2)

            l = detector.findDistance(lmList1[8], lmList1[12], img)[0]
            print(l)

        else:
            l = detector.findDistance(lmList1[8], lmList1[12], img)[0]
            print(l)

        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < hands[0]['lmList'][8][0] < x + w and y < hands[0]['lmList'][8][1] < y + h:
                cv.rectangle(img, button.pos, (x+w, y + h), (175, 0, 175), cv.FILLED)
                cv.putText(img, button.text, (x + 25, y + 65), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
            
                if l < 30:
                    keyboard.press(button.text)
                    cv.rectangle(img, button.pos, (x+w, y + h), (0, 255, 0), cv.FILLED)
                    cv.putText(img, button.text, (x + 25, y + 65), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                    finalText += button.text
                    sleep(0.15)

    cv.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv.FILLED)
    cv.putText(img, finalText, (60, 430), cv.FONT_HERSHEY_PLAIN, 5, (255, 255, 255 ), 5)



    cv.imshow("Image", img)
    if cv.waitKey(1) == ord('q'):  
        break

cap.release()
cv.destroyAllWindows()
