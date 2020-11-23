# usage: python3 convert_tflite.py b0_model.tflite
# abdadoned

from rknn.api import RKNN
import sys


if len(sys.argv) < 2:
    print('At least two argument: tflite input')
    exit(1)

model_tflite = sys.argv[1]
if '.tflite' not in model_tflite:
    print('The given input is not tflite file')
    exit(2)

# Create RKNN object
rknn = RKNN()

# pre-process config
print('--> config model')
rknn.config(channel_mean_value='127.5 127.5 127.5 127.5', reorder_channel='0 1 2')
print('Configured')

# Load tensorflow model
print('--> Loading model')
ret = rknn.load_tflite(model=model_tflite)
if ret != 0:
    print('Load {} failed!'.format(model_tflite))
    exit(ret)
print('Loaded')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=True, dataset='./dataset.txt', pre_compile=False)
if ret != 0:
    print('Build {} failed!'.format(model_tflite))
    exit(ret)
print('Building completed!')

model_rknn = model_tflite.replace('.tflite', '.rknn')
# Export rknn model
print('--> Export RKNN model')
ret = rknn.export_rknn(model_rknn)
if ret != 0:
    print('Export {} failed!'.format(model_rknn))
    exit(ret)
print('Export completed!')


rknn.release()
