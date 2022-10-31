import numpy as np
import torch
import cv2

from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression

from src import model

from src.body import Body
from src.hand import Hand

imgsz = (640, 640)
conf_thres = 0.6
iou_thres = 0.5
classes = None
agnostic_nms = 2
max_det = 2
device = select_device("")
body_estimation=""

def launch_yolo_model(yPath):
	model = DetectMultiBackend(yPath, device=device, dnn=False, data="./data/baseball_dataset.yaml", fp16=False)
	stride, names, pt = model.stride, model.names, model.pt
	model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))

	return model

def pram_init(oPath):
	global body_estimation
	body_estimation = Body(oPath)



def IoU(box1, box2):
    Area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    Area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    if ( min(box1[2], box2[2]) - max(box1[0], box2[0]) < 0) or (min(box1[3], box2[3]) - max(box1[1], box2[1])  <0 ):
        return 0 
    Intersection =(min(box1[2], box2[2]) - max(box1[0], box2[0]))*(min(box1[3], box2[3]) - max(box1[1], box2[1]))
    #Union = Area1 + Area2 - Intersection
    Union = Area1
    return Intersection/Union

def yoloPred(frame, yolo_model):
	im = frame.copy()
	im = letterbox(im, imgsz, 32, True)[0]
	im = im.transpose((2, 0, 1))[::-1]
	im = np.ascontiguousarray(im)
	im = torch.from_numpy(im).to(device)
	im = im.half() if yolo_model.fp16 else im.float()
	im /= 255
	if len(im.shape) == 3:im = im[None]
	pred = yolo_model(im, augment=False, visualize=False)
	pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

	pred = pred[0].tolist()
	for i in range(len(pred)):
		for j in range(len(pred[i])):
			pred[i][j] = int(pred[i][j])

	return pred

def poenposePred(frame):
	oriImg = frame.copy()
	candidate, subset = body_estimation(oriImg)

	return candidate, subset

def paintPolyline(frame, pointList):
	if len(pointList)>0:
		points = np.array(pointList, np.int32)
		frame = cv2.polylines(frame, pts=[points], isClosed=False, color=(255, 0, 0), thickness=2)

	return frame

def cap2list(cap):
	vList=[]
	f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	for i in range(f_count):
		ret, frame = cap.read()
		frame = cv2.resize(frame, (382, 640))
		vList.append(frame)
	return vList

def inBox(p, sBox):
	if sBox[0]<p[0] and p[0]<sBox[2] and sBox[1]<p[1] and p[1]<sBox[3]:
		return True
	else:
		return False


def detectStrike(pList, sBox):
	for i in range(len(pList)-1):
		line = [pList[i], pList[i+1]]
		v = [line[1][0]-line[0][0], line[1][1]-line[0][1]]
		nv = [v[0]//3, v[1]//3]
		for i in range(5):
			nline = [line[0][0]+(nv[0]*i), line[0][1]+(nv[1]*i)]
			if inBox(nline, sBox):
				return True

	return False
			
def startRecording(frame, yolo_model):
	Pred = yoloPred(frame, yolo_model)
	head_center = [0, 0]
	glove_center = [0, 0]

	for i in range(len(Pred)):
		if Pred[i][5]==0:
			head_center = [Pred[i][0]+((Pred[i][2]-Pred[i][0])/2), Pred[i][1]+((Pred[i][3]-Pred[i][1])/2)]
		if Pred[i][5]==1:
			glove_center = [Pred[i][0]+((Pred[i][2]-Pred[i][0])/2), Pred[i][1]+((Pred[i][3]-Pred[i][1])/2)]

	if glove_center[1]<head_center[1]:
		return True
	else:
		return False
