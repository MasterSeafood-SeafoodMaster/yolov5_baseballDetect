import cv2
import os
import threading
import time
import argparse
import utils.baseball_toolkit as bt
import numpy as np
from src import util

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#------------------------threading------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--videoPath', type=str, default="./testing_data/behind_11.mp4")
parser.add_argument('--type', type=str, default="behind")
args = parser.parse_args()
video_path = args.videoPath
videoType = args.type

bt.pram_init('./custom_model/body_pose_model.pth')
noballDetected=0
ballTrack=[]
frame=""
tLock = threading.Lock()
class yoloThread():
	def __init__(self, vList, yolo_model):
		self.yolo_model = yolo_model
		self.switch = True
		self.vList = vList
		self.ballTrack = []
		self.homePlate = []
		self.noballDetected = 0
		self.result=[]
		self.maxPath=[]
		self.thread=threading.Thread(target=self.update, args=())
		self.thread.daemon=True
		self.thread.start()

	def update(self):
		for f in range(len(self.vList)):
			self.frame = self.vList[f]
			self.yoloPred = bt.yoloPred(self.frame, self.yolo_model)
			for i in range(len(self.yoloPred)):
				if self.yoloPred[i][5]==0:
					self.ballTrack.append([ (self.yoloPred[i][0]+self.yoloPred[i][2])//2, (self.yoloPred[i][1]+self.yoloPred[i][3])//2 ])
					self.noballDetected=0
				else:
					if self.noballDetected>10:
						if self.maxPath==[]:
							self.maxPath = self.ballTrack.copy()
						self.ballTrack=[]
					self.noballDetected+=1

				if self.yoloPred[i][5]==1:
					self.homePlate.append([self.yoloPred[i][0], self.yoloPred[i][1], self.yoloPred[i][2], self.yoloPred[i][3]])

			self.result.append(self.ballTrack.copy())
		self.switch = False

	def getReturn(self):
		if self.switch:
			return "still_working"
		else:
			return [self.result, self.maxPath, self.homePlate] #[ball path per frame, longest path]

class openposeThread():
	def __init__(self, vList):
		self.switch = True
		self.vList = vList
		self.cResult=[]
		self.sResult=[]
		self.thread=threading.Thread(target=self.update, args=())
		self.thread.daemon=True
		self.thread.start()
	def update(self):
		for f in range(len(self.vList)):
			if f%50==0:
				self.frame = self.vList[f]
				
				self.candidate, self.subset = bt.poenposePred(self.frame)
				self.cResult.append(self.candidate)
				self.sResult.append(self.subset)
		self.switch = False
		
	def getReturn(self):
		if self.switch:
			return "still_working"
		else:
			fail=0
			l = len(self.cResult)
			self.strikeBox=[0, 0, 0, 0]
			for i in range(l):
				try:
					self.strikeBox[0]+=self.cResult[i][3][0]
					self.strikeBox[1]+=self.cResult[i][3][1]
					self.strikeBox[2]+=self.cResult[i][12][0]
					self.strikeBox[3]+=self.cResult[i][12][1]
				except:
					fail+=1
			for i in range(len(self.strikeBox)):
				self.strikeBox[i] = self.strikeBox[i]//(l-fail)
				self.strikeBox[i] = int(self.strikeBox[i])

			return [self.cResult, self.sResult, self.strikeBox] #[candidate, subset, average strike zone]

#------------------------main------------------------

def BRS(cap, videoType, visualize, yolo_model):
	vList = bt.cap2list(cap)

	if videoType=="behind":
		yThread = yoloThread(vList, yolo_model)
		yList=yThread.getReturn()
		while (yList=="still_working"):
			yList=yThread.getReturn()
			time.sleep(0.5)

		r = np.array(yList[2])
		plate = np.mean(r, axis=0)
		
		Strike = bt.detectStrike(yList[1], plate) 
		#print(Strike)
		
	elif videoType=="side":
		yThread = yoloThread(vList, yolo_model)
		oThread = openposeThread(vList)

		oList=oThread.getReturn()
		yList=yThread.getReturn()

		while (oList=="still_working" or yList=="still_working"):
			yList=yThread.getReturn()
			oList=oThread.getReturn()
			time.sleep(0.5)



		Strike = bt.detectStrike(yList[1], oList[2]) 
		#print(Strike)

	if visualize:
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (382,  640))

		#------------------------visualize------------------------

		for i in range(len(vList)):
			frame = vList[i]
			#print(frame.shape) 
			frame = bt.paintPolyline(frame, yList[0][i])

			if videoType=="side":
				frame = util.draw_bodypose(frame, oList[0][i//50], oList[1][i//50])
				frame = cv2.rectangle(frame, (oList[2][0], oList[2][1]), (oList[2][2], oList[2][3]), (0, 0, 255), 2)
			elif videoType=="behind":
				frame = cv2.rectangle(frame, (int(plate[0]), int(plate[1])), (int(plate[2]), int(plate[3])), (0, 0, 255), 2)

			cv2.imshow('live', frame)
			if cv2.waitKey(50) == ord('q'):
				break
			out.write(frame)

		out.release()
		cv2.destroyAllWindows()
	return Strike