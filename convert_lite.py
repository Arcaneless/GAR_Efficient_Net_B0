from rknn.api import RKNN
import numpy as np
import sys
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("TTLite file required")
        exit(1)
    model_name = sys.argv[1]

    # Create RKNN object
    rknn = RKNN(verbose=1)
    
    # pre-process config
    print('--> config model')
    rknn.config(batch_size=50, channel_mean_value='0 0 0 1', reorder_channel='0 1 2')
    print('done')

    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_tflite(model=model_name)
    if ret != 0:
        print('Load efficient net failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset_large.txt')
    if ret != 0:
        print('Build efficient net failed!')
        exit(ret)
    print('done')

    # rknn.accuracy_analysis(inputs='./dataset_2.txt', target='RK3399PRO', calc_qnt_error=True)

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(model_name.replace('.tflite', '.rknn'))
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')
