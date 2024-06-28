import cv2 as cv
import  mediapipe as mp
import time
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec
landmark_drawing_spec = DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)  # Green color for landmarks
connection_drawing_spec = DrawingSpec(color=(0, 255, 0), thickness=2)  # Red color for connections

class handDetector():
    
    def __init__(self, mode=False, maxHands=2, complexity = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.tipIds = [4, 8 ,12, 16, 20]

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,self.detectionCon, self.trackCon,)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw =True):
        imgRGB =cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)  ------> tells coordinates or landmarks

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS,landmark_drawing_spec,connection_drawing_spec)
        return img

    def findPosition(self, img,handNo=0,draw=True):
            self.lmList=[]
            if self.results.multi_hand_landmarks:
                myHand = self.results  .multi_hand_landmarks[handNo]
                for id,lm in enumerate(myHand.landmark):  # lm containes x,y,z coordinates of landmark points
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w),int(lm.y*h)
                    # print(id,cx,cy)
                    self.lmList.append([id,cx,cy])
                    if draw:
                        cv.circle(img,(cx,cy),3,(255,0,255),cv.FILLED)
            return self.lmList
      
    def fingerUp(self):
            
            fingers = []
            #thumb

            
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]: 
                    fingers.append(1)
            else:
                    fingers.append(0)

            for id in range(1,5):
                
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            return fingers

    


    


def main():
    cTime=0
    pTime =0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img,draw=False) 
        if len(lmList) != 0:
            print(lmList)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_DUPLEX,3,(20,255,57),3)

        cv.imshow("Image",img)
        cv.waitKey(15)


if __name__ == "__main__":
    main()