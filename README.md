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
