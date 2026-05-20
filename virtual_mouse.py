import cv2
import numpy as np
import time
import HandTracking as ht
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# ===============================
# FAST VARIABLES
# ===============================
width, height = 640, 480
frameR = 40              # more usable space
smoothening = 4          # faster response

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

last_click_time = 0
click_delay = 0.25

prev_scroll_y = None
prev_vol_y = None

# Camera
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = ht.handDetector(maxHands=1)

screen_width, screen_height = pyautogui.size()

pTime = 0

# ===============================
# MAIN LOOP
# ===============================
while True:
    success, img = cap.read()
    if not success:
        continue

    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img)

    if len(lmlist) != 0:

        x1, y1 = lmlist[8][1:]   # Index
        x2, y2 = lmlist[12][1:]  # Middle

        fingers = detector.fingersUp()

        # Draw region
        cv2.rectangle(img, (frameR, frameR),
                      (width-frameR, height-frameR),
                      (255, 0, 255), 2)

        # ===============================
        # 1. FAST CURSOR MOVE
        # ===============================
        if fingers[1] == 1 and fingers[2] == 0:

            x3 = np.interp(x1, (frameR, width-frameR), (0, screen_width))
            y3 = np.interp(y1, (frameR, height-frameR), (0, screen_height))

            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            pyautogui.moveTo(screen_width - curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y
            prev_scroll_y = None
            prev_vol_y = None

        # ===============================
        # 2. LEFT CLICK
        # ===============================
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:

            length, _, _ = detector.findDistance(8, 12, img)

            if length < 30:
                if time.time() - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = time.time()

            prev_scroll_y = None
            prev_vol_y = None

        # ===============================
        # 3. RIGHT CLICK
        # ===============================
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:

            if time.time() - last_click_time > click_delay:
                pyautogui.rightClick()
                last_click_time = time.time()

            prev_scroll_y = None
            prev_vol_y = None

        # ===============================
        # 4. SMOOTH SCROLL
        # ===============================
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:

            if prev_scroll_y is not None:
                diff = y2 - prev_scroll_y

                if abs(diff) > 5:
                    pyautogui.scroll(int(-diff * 2))

            prev_scroll_y = y2
            prev_vol_y = None

        # ===============================
        # 5. VOLUME CONTROL (FAST)
        # ===============================
        elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0:

            if prev_vol_y is not None:
                diff = y1 - prev_vol_y

                if abs(diff) > 8:
                    pyautogui.scroll(int(-diff * 3))  # acts like volume

            prev_vol_y = y1
            prev_scroll_y = None

        else:
            prev_scroll_y = None
            prev_vol_y = None

    # ===============================
    # FPS
    # ===============================
    cTime = time.time()
    fps = int(1 / (cTime - pTime)) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {fps}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()