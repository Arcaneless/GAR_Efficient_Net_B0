from rknn.api import RKNN
import numpy as np
import cv2
import pandas as pd

imgs = ['0001.jpg', '0002.jpg', '0003.jpg', '0004.jpg', '0005.jpg', '0006.jpg', '0007.jpg', '0008.jpg', '0009.jpg', '0010.jpg']
imgs = [cv2.imread('train/{}'.format(x)) for x in imgs]
imgs = [cv2.resize(x, (224,224)) for x in imgs]
imgs = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in imgs]
imgs = np.array(imgs).astype(np.uint8)

rknn = RKNN()
rknn.load_rknn('b0_model_lite.rknn')
# init runtime environment
print('--> Init runtime environment')
ret = rknn.init_runtime()
if ret != 0:
    print('Init runtime environment failed')
    exit(ret)
print('done')

# Inference
print('--> Running model')
result = []
for img in imgs:
    outputs = rknn.inference(inputs=[img])
    outputs = [np.argmax(x) for x in outputs]
    print(outputs)
    result.append(outputs)
print('done')
test_frame = pd.DataFrame(np.array(result))
print(test_frame)
real_frame = pd.read_csv('train/train.csv', header=None)
print(real_frame)


rknn.release()
