from rknn.api import RKNN
import numpy as np
import cv2
import pandas as pd

imgs = pd.read_csv('dataset_large.txt',header=None)
imgs = imgs.iloc[:,0]
imgs = imgs.tolist()
imgs = [cv2.imread(x) for x in imgs]
imgs = [cv2.resize(x, (224,224)) for x in imgs]
imgs = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in imgs]

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

with open('output.txt', 'w') as f:
    for name in result:
        f.write('{}\n'.format(name))
       
rknn.release()
