import cv2  # Can be installed using "pip install opencv-python"
import mediapipe as mp  # Can be installed using "pip install mediapipe"
import time
import math
import numpy as np
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, modelComplexity=1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.tipIds = [4, 8, 12, 16, 20]
        self.timestamp_ms = 0
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "hand_landmarker.task")
        if not os.path.exists(self.model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_hand_presence_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True):    # Finds all hands in a frame
        if img is None:
            self.results = None
            return img
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.timestamp_ms += 33
        self.results = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        return img

    def findPosition(self, img, handNo=0, draw=True, drawBbox=False):   # Fetches the position of hands
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if img is None:
            return self.lmList, bbox
        h, w, c = img.shape
        if self.results and self.results.hand_landmarks:
            hands = self.results.hand_landmarks
            if handNo < len(hands):
                hand = hands[handNo]
                # Pre-calculate cx, cy to avoid redundant operations
                for id, lm in enumerate(hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                
                if draw:
                    # Hand skeleton connections based on the provided image
                    connections = [
                        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),  # Palm base to finger MCPs
                        (1, 2), (2, 3), (3, 4),                     # Thumb
                        (5, 6), (6, 7), (7, 8),                     # Index
                        (9, 10), (10, 11), (11, 12),                # Middle
                        (13, 14), (14, 15), (15, 16),               # Ring
                        (17, 18), (18, 19), (19, 20),               # Pinky
                        (5, 9), (9, 13), (13, 17)                   # MCP to MCP palm connections
                    ]
                    
                    # Draw green lines first (skeleton)
                    for start_idx, end_idx in connections:
                        x1, y1 = self.lmList[start_idx][1], self.lmList[start_idx][2]
                        x2, y2 = self.lmList[end_idx][1], self.lmList[end_idx][2]
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw red circles on top (landmarks)
                    for id, cx, cy in self.lmList:
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                
                if drawBbox:
                    xList = [lm[1] for lm in self.lmList]
                    yList = [lm[2] for lm in self.lmList]
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    bbox = xmin, ymin, xmax, ymax
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                  (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self):    # Checks which fingers are up
        if not self.lmList or len(self.lmList) < 21:
            return []
        fingers = []
        # Thumb: check if tip is to the right/left of IP joint (depending on hand side)
        # Simplified: check x position relative to IP joint
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers: check if tip is above pip joint (y-coordinate is smaller)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):   # Finds distance between two fingers
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success:
            continue
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
