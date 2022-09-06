import numpy as np
import cv2
import time
from PIL import ImageGrab
import pytesseract
import pyautogui

pytesseract.pytesseract.tesseract_cmd = "F:\Program Files\Tesseract-OCR\Tesseract.exe"

prices = []

def startGame():
        #start the game, giving the user a few seconds to click on the chrome tab after starting the code
    for i in reversed(range(4)):
        print("agent starting in ", i)
        time.sleep(1)

def activity():

    #bbox=(left_x, top_y, right_x, bottom_y)


    screen2 = np.array(ImageGrab.grab(bbox=(650, 340, 800, 365)))
    grey2 = cv2.cvtColor(screen2, cv2.COLOR_BGR2GRAY)
    score1 = pytesseract.image_to_string(grey2)
    print(score1)
    cv2.imshow("image", np.array(grey2))
    cv2.waitKey(0)

startGame()
activity()
