# GAR_Efficient_Net_B0
  
## Usage
### `convert_lite.py`:
This is the convertion program from tflite to rknn with quantization
`python3 convert_lite.py b0_model_lite.tflite`
#### Output
`b0_model_lite.rknn`

### `build_dataset.py`:
Turn the `lfw_names.txt` to usable `dataset_large.txt`
