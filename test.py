import cv2
from videoDetect import BRS

cap = cv2.VideoCapture("./testing_data/behind_11.mp4")
Strike = BRS(cap, "behind", True) #visualize?
print(Strike)