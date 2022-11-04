import cv2
import utils.baseball_toolkit as bt
from videoDetect import BRS

baseball_model = bt.launch_yolo_model("./custom_model/baseball_model.pt")

cap = cv2.VideoCapture("./testimg/behind_15.mp4")
Strike_behind = BRS(cap, "behind", True, baseball_model)

cap = cv2.VideoCapture("./testimg/side_15.mp4")
Strike_side = BRS(cap, "side", True, baseball_model)

if Strike_behind and Strike_side:
	print(True)
else:
	print(False)


#bt.startRecording(frame, hg_model) 
#BRS(cap, "behind", True, baseball_model)