# rk3566NpuModel
This repo has code to create a matmul model in torch then convert it to rknn model and run it on rockchip npu

Story time 
following my adventures with Raspberry Pi Zero 2W and OpenCL, I started my search for another SBC with Zero form factor that can perform AI/ML operation at fast enough rate.
After search for a while, I got myself the Radxa Zero 3W board (4GB model)
It has a quad-core ARM CPU (Good)
Mali-52 GPU that supports Vulkan and OpenCL (Not really but thats a story for another day)
and most importantly it has a built in NPU that can do 1TOPS@INT8.
Radxa Zero 3W comes in with either 1GB / 2GB / 4GB / 8GB LPDDR4.
my understanding is that the NPU comes disabled by default, because while it is part of the rk3566 chip is not verified by Radxa, you need to enable it manually and hope that it works. I did so on my board and i think it works :D 

The good folks at QEngineering have a repo with an Ubuntu Image with everything already installed and the NPU enabled , their Wiki also explains how to enable it yourself https://github.com/Qengineering/Radxa-Zero-3-NPU-Ubuntu22/wiki 

## Options to get MatMul working on the NPU
My plan was to run a couple of matrices operations on the NPU to see how it behaves and just mess around. and my understanding was that Rockchip provides an SDK that you can use in Python and C. 
After a few days of research i discovered that the SDK Python APIs can only be used to just run models, while C APIs can be used to run matmul.

As a workaround to run Matmul with Python on the NPU, we could either use the C code inside python with ctypes(looks weird), or create a model with a single layer of matmul then run it on the NPU. I opted for the latter
## How to run MatMul
So the Plan to run MatMul is Simple :
Create Pytorch model with a single matmul layer >> export as onnx >> use RKNN toolkit to convert onnx model to Rknn model >> run model on NPU

RKnn Toolkit doesnt work on ARM so we need x86-64 PC, I made a quick google colab notebook (RK3566_NPU_Model.ipynb) that installs rknn-toolkit, creates the model and convert it to rknn model

Note that when we create the model we should specify that random tensors type to be float16 because the Rk3566 npu only support fp16 and the tensors type created by torch were float32 by default,
In reality that doesn't matter because rknn runs at FP16 anyway even if the model was fp32

Run npuMatmulModel.py 

Expected output with Python 3.10.12
```python
python3 npuMatmulModel.py 
W rknn-toolkit-lite2 version: 2.3.2
I RKNN: [22:16:58.398] RKNN Runtime Information, librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
I RKNN: [22:16:58.399] RKNN Driver Information, version: 0.9.6
I RKNN: [22:16:58.399] RKNN Model Information, version: 6, toolkit version: 2.3.2(compiler version: 2.3.2 (e045de294f@2025-04-07T19:48:25)), target: RKNPU lite, target platform: rk3566, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [22:16:58.404] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_shapes
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning message.)
Matrix 1:
[[0.4934  0.593   0.8374  0.646   0.185   0.9985  0.1071  0.558   0.174  ]
 [0.326   0.559   0.452   0.8706  0.569   0.883   0.1854  0.2874  0.3298 ]
 [0.38    0.5083  0.5215  0.0937  0.067   0.8047  0.1996  0.1321  0.312  ]
 [0.944   0.752   0.0684  0.702   0.03464 0.942   0.711   0.135   0.804  ]
 [0.38    0.6245  0.1705  0.97    0.8877  0.5674  0.7896  0.3716  0.399  ]
 [0.2262  0.354   0.1283  0.9136  0.01645 0.4502  0.4204  0.7827  0.2234 ]
 [0.4224  0.7773  0.6304  0.6587  0.6714  0.842   0.9106  0.9917  0.1467 ]
 [0.9863  0.977   0.8696  0.518   0.4026  0.0738  0.9097  0.9067  0.7974 ]
 [0.134   0.854   0.0704  0.3423  0.528   0.8887  0.533   0.4807  0.747  ]]

Matrix 2:
[[0.1197   0.2664   0.4314   0.959    0.945    0.6177   0.07947  0.3281
  0.5728  ]
 [0.8115   0.9165   0.852    0.1404   0.8413   0.4426   0.3376   0.3171
  0.02785 ]
 [0.8345   0.1714   0.5684   0.7305   0.2795   0.491    0.3496   0.9756
  0.903   ]
 [0.449    0.79     0.4172   0.6357   0.7803   0.5186   0.4255   0.581
  0.664   ]
 [0.9326   0.6562   0.721    0.2167   0.4905   0.009544 0.46     0.5156
  0.1254  ]
 [0.181    0.3635   0.265    0.1509   0.776    0.2854   0.5127   0.633
  0.4575  ]
 [0.4546   0.3403   0.958    0.5186   0.397    0.953    0.3289   0.404
  0.396   ]
 [0.6777   0.9766   0.508    0.002102 0.2463   0.01982  0.628    0.0977
  0.1483  ]
 [0.4473   0.6763   0.4521   0.372    0.286    0.535    0.603    0.5903
  0.421   ]]

Resulting matrix (Matrix 1 × Matrix 2):
[[2.387 2.512 2.326 1.891 2.799 1.807 1.895 2.471 2.162]
 [2.377 2.625 2.354 1.751 2.787 1.738 1.897 2.381 1.919]
 [1.463 1.475 1.594 1.232 1.863 1.354 1.204 1.671 1.371]
 [2.072 2.789 2.768 2.324 3.385 2.693 1.954 2.412 2.164]
 [2.85  3.16  3.113 2.029 3.084 2.242 2.146 2.461 1.939]
 [1.75  2.361 1.886 1.315 2.041 1.497 1.574 1.514 1.446]
 [3.434 3.578 3.627 2.195 3.373 2.469 2.547 2.812 2.309]
 [3.643 3.742 3.971 2.916 3.469 3.074 2.506 2.965 2.637]
 [2.477 2.926 2.678 1.321 2.623 1.849 2.096 2.121 1.461]]

NumPy result (for verification):
[[2.387 2.512 2.326 1.891 2.799 1.807 1.895 2.47  2.162]
 [2.377 2.625 2.354 1.751 2.787 1.738 1.897 2.38  1.919]
 [1.463 1.475 1.594 1.232 1.863 1.354 1.204 1.671 1.371]
 [2.072 2.79  2.768 2.324 3.385 2.693 1.954 2.412 2.164]
 [2.85  3.16  3.113 2.03  3.084 2.242 2.146 2.46  1.939]
 [1.75  2.361 1.886 1.315 2.041 1.497 1.574 1.514 1.446]
 [3.434 3.578 3.627 2.195 3.373 2.469 2.547 2.812 2.309]
 [3.643 3.742 3.97  2.916 3.469 3.074 2.506 2.965 2.637]
 [2.477 2.926 2.678 1.321 2.623 1.849 2.096 2.121 1.461]]
```
