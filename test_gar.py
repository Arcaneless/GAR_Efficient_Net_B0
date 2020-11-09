# usage: python3 test_gar.py b0_model.rknn

from rknn.api import RKNN
import numpy as np
import cv2

if len(sys.argv) < 2:
    print('At least two argument: rknn input')
    exit(1)

model_rknn = sys.argv[1]
if '.rknn' not in model_rknn:
    print('The given input is not rknn file')
    exit(2)

rknn = RKNN()
rknn.load_rknn(model_rknn)

img = cv2.imread('./train/0001.jpg')
img = cv2.resize(img, (224,224))
outputs = rknn.inference(inputs=[img])
print(outputs)

rknn.release()