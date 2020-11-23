# GAR_Efficient_Net_B0
  
## Usage
### Large dataset
Download the `lfw.tgz` from <http://vis-www.cs.umass.edu/lfw/lfw.tgz>.
Unzip it. (`tar -zxvf lfw.tgz`)

### `convert_lite.py`:
This is the convertion program from tflite to rknn with quantization.
It takes the input and use `dataset_large.txt` to quantize the model.
e.g. `python3 convert_lite.py b0_model_lite.tflite`
#### Output
`b0_model_lite.rknn`

### `build_dataset.py`:
Turn the `lfw_names.txt` to usable `dataset_large.txt`.
#### Output
`dataset_large.txt`

### `real_test.py`:
Test the rknn model (hardcoded b0_model_lite.rknn currently) using the given large dataset.
e.g. `python3 real_test.py`
#### Output
The stdout of the inference as expected

### `EfficientNet_GAR_Test.ipynb`:
The making of the tflite model. (see the comment inside)
Input: the image with size (224,224,3)
Output: gender, age and race
