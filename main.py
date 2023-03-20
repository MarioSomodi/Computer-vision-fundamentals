import os
import torch
from collections import Counter
import cv2
import numpy as np
from matplotlib import pyplot as plt
from FoundObject import FoundObject


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

objectNamesAndCounts = {}
for pathToImage in absoluteFilePaths("images/dogs"):
    results = model(pathToImage)
    results.render()
    foundObjectsNames = results.pandas().xyxy[0]["name"]
    foundObjectsConfidence = results.pandas().xyxy[0]["confidence"]
    foundObjects = []
    for n in range(0, len(foundObjectsNames)):
        foundObjects.append(FoundObject(foundObjectsNames.values[n], float(foundObjectsConfidence.values[n])))
    for object in foundObjects:
        if (object.confidence > 0.7):
            if (object.name in objectNamesAndCounts):
                objectNamesAndCounts[object.name] += 1
            else:
                objectNamesAndCounts[object.name] = 1
print(objectNamesAndCounts)
