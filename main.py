import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Load image
img = cv2.imread('images/example.jpg')
# Inference
results = model(img)
# Plot results
results.render() # or results.show()
# Save results
results.save('output.jpg')