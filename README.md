# Object Detection Using YOLOv3 in Google Colab

This project demonstrates object detection on images using the **YOLOv3** model and OpenCV. YOLOv3 identifies objects in an uploaded image and highlights them with bounding boxes and confidence scores.

---

## Table of Contents
1. [Overview](#overview)
2. [Technologies Used](#technologies-used)
3. [Setup Instructions](#setup-instructions)
4. [Output](#output)
5. [Future Enhancements](#future-enhancements)
6. [Contact](#contact)

---

## Overview

**YOLOv3** (You Only Look Once v3) is a state-of-the-art, real-time object detection system. This project uses the pre-trained YOLOv3 model trained on the COCO dataset for object detection.

---

## Technologies Used

- Python
- OpenCV
- Google Colab
- NumPy
- Matplotlib

---

## Setup Instructions

Run the following commands in your Google Colab notebook to set up the environment, install required packages, download YOLOv3 pretrained files, and upload an image:

```python
# Step 1: Install Required Packages
!pip install opencv-python
!pip install opencv-python-headless

# Step 2: Download YOLOv3 Pretrained Files
!wget https://pjreddie.com/media/files/yolov3.weights
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# Step 3: Import Necessary Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 4: Load YOLO Network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Step 5: Load COCO Class Labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Step 6: Upload an Image
from google.colab import files
uploaded = files.upload()
input_image_name = list(uploaded.keys())[0]

# Step 7: Preprocess the Image for YOLO
img = cv2.imread(input_image_name)
height, width, channels = img.shape
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Step 8: Perform Object Detection
boxes = []
confidences = []
class_ids = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Step 9: Draw Bounding Boxes on the Image
for i in indices:
    box = boxes[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

# Step 10: Display the Output Image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()

```
## Output: Before and After Object Detection

Here are the before and after images side by side:

<table>
  <tr>
    <td><strong>Before:</strong><br><img src="https://github.com/yashpatel0110/Object-Detection/blob/main/before.jpg" width="300" /></td>
    <td><strong>After:</strong><br><img src="https://github.com/yashpatel0110/Object-Detection/blob/main/after.png" width="300" /></td>
  </tr>
</table>

