# usage: python3 test_rknn.py b0_model.rknn


from rknn.api import RKNN

if len(sys.argv) < 2:
    print('At least two argument: rknn input')
    exit(1)

model_rknn = sys.argv[1]
if '.rknn' not in model_rknn:
    print('The given input is not rknn file')
    exit(2)

rknn = RKNN()
rknn.load_rknn(model_rknn)

print('--> Init runtime environment')
ret = rknn.init_runtime()
if ret != 0:
    print('Init runtime environment failed')
    exit(ret)
print('done')
print('--> Begin evaluate model performance')
perf_results = rknn.eval_perf(inputs=None)
print('done')

rknn.release()