import pandas as pd

frame = pd.read_csv('dataset.txt', header=None)
arr = frame.iloc[:,0]
print(arr.to_list())

with open('dataset_repeated.txt', 'w') as f:
    for i in range(100):
        for txt in arr:
            f.write('{}\n'.format(txt))
