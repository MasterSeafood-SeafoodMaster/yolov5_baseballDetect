import cv2
import utils.baseball_toolkit as bt
from videoDetect import BRS

baseball_model = bt.launch_yolo_model("./custom_model/baseball_model.pt")
hg_model = bt.launch_yolo_model("./custom_model/hg_model.pt")

cap = cv2.VideoCapture("./testing_data/test01.mp4")

f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(f_count):
	ret, frame = cap.read()

	isRecord, frame = bt.startRecording(frame, hg_model)
	print(isRecord)

	cv2.imshow("live", frame)
	cv2.waitKey(1)

cap = cv2.VideoCapture("./testing_data/test01.mp4")
Strike = BRS(cap, "behind", True, baseball_model)
print(Strike)

#bt.startRecording(frame, hg_model) 
#BRS(cap, "behind", True, baseball_model)