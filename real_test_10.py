from rknn.api import RKNN
import numpy as np
import cv2
import pandas as pd
import sys
if len(sys.argv) < 2:
    print('RKNN file required')
    print('usage: python3 real_test.py \'filename\'')
    exit(1)

model_name = sys.argv[1]

names = ['0001.jpg', '0002.jpg', '0003.jpg', '0004.jpg', '0005.jpg', '0006.jpg', '0007.jpg', '0008.jpg', '0009.jpg', '0010.jpg']
imgs = ['train/{}'.format(x) for x in names]
filenames = imgs
imgs = [cv2.imread(x) for x in imgs]
imgs = [cv2.resize(x, (224,224)) for x in imgs]
imgs = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in imgs]

rknn = RKNN()
rknn.load_rknn(model_name)
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
for i, img in enumerate(imgs):
    outputs = rknn.inference(inputs=[img])
    outputs = [np.argmax(x) for x in outputs]
    outputs[0], outputs[1] = outputs[1], outputs[0]
    print(filenames[i], outputs)
    result.append(outputs)
print('done')
test_frame = pd.DataFrame(np.array(list(zip(filenames,result))))
print(test_frame)
test_frame.to_csv('output.csv')

with open('output.txt', 'w') as f:
    for name in result:
        f.write('{}\n'.format(name))

rknn.release()
print('Ground Truth')
real_frame = pd.read_csv('train/train.csv', header=None)
print(real_frame)
