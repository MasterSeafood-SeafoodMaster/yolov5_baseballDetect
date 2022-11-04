# yolov5_baseballDetect

## Goal
We design a **Baseball Recognition System** and **pack it as a library**. So that you could use it by just importing it as a package.

---
## Pre-Requisites
1. Open anaconda and activate your envirobment.
	```bash
	conda activate your_env
	```
2. Open the baseball system directory.
	```bash
	cd ./yolov5_baseball
	```
3. Install the required packages by type this command below: 
	```bash
	pip install -r requirements.txt
	```
4. Install scikit-image using `pip/pip3`
	```bash
	pip install scikit-image
	```
5. Download the custom models from [google drive](https://drive.google.com/drive/folders/181GHT1pYWCMIk7TnV-kF2gJYnE5KEFVT?usp=sharing) and then put it to `./custom_model`.

---

## Dataset collection
1. COCO Dataset [download here](https://cocodataset.org/#home)
2. Baseball Dataset `./data/baseball_dataset.yaml` from Baseball Strike and Recognition System [^2]

---

## The flow of baseball detection system
1. To detect the baseball and the home plate using YOLOv5 [^1].
2. Draw the tracking line of the ball. (for easier visualization)
3. To detect the skeleton of the human, then calculate and frame out the strike zone by the knees and elbows of the batter by OpenPose [^4].
4. If the baseball goes through the home plate and the strike zone, then recognize this ball as a strike (good ball). Otherwise, mark it as a bad ball.

---

## How to use it?
Use the BRS system as a function `BRS()` by importing it as a package. Here is a sample code that demonstrates how to use this library: 
```python
import cv2
import utils.baseball_toolkit as bt
from videoDetect import BRS

baseball_model = bt.launch_yolo_model("./custom_model/baseball_model.pt")

cap = cv2.VideoCapture("./testing_data/test01.mp4")
Strike = BRS(cap, "behind", True, baseball_model)
print(Strike)
```

---

## Demo
https://github.com/MasterSeafood-SeafoodMaster/yolov5_baseballDetect/blob/main/img/DEMO_VIDEO.mp4

---

## Reference
[^1]: *YOLOv5: https://github.com/ultralytics/yolov5*
[^2]: *Yi-Ping Chen. "A Baseball Strike and Ball Recognition System Based on Deep Learning." 2022*
[^3]: *Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "Yolov4: Optimal speed and accuracy of object detection." arXiv preprint arXiv:2004.10934 (2020).*
[^4]: *Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei and Yaser Sheikh. "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields." IEEE Transactions on Pattern Analysis and Machine Intelligence arXiv:1812.08008 (2019)*
[^5]: *Tomas Simon, Hanbyul Joo, Iain Matthews and Yaser Sheikh. "Hand Keypoint Detection in Single Images using Multiview Bootstrapping." CVPR (2017) arXiv:1704.07809 (2017)*
[^6]: *Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh. "Convolutional pose machines" arXiv:1602.00134 (2016)*
[^7]: *Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick and Piotr Dollar. "Microsoft COCO: Common Objects in Context." arXiv:1405.0312v3 (2015)*
