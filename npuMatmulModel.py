from rknnlite.api import RKNNLite
import numpy as np

# Initialize RKNNLite
rknn_lite = RKNNLite()

# Load the RKNN model
ret = rknn_lite.load_rknn('matmul.rknn')
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)

# Initialize the runtime environment
ret = rknn_lite.init_runtime()
if ret != 0:
    print('Init runtime failed')
    exit(ret)

#Create Two 9x9 matrices with batch size of 1
matrix1 = np.random.rand(1, 9, 9).astype(np.float16)
matrix2 = np.random.rand(1, 9, 9).astype(np.float16)

# Perform inference using the NPU
outputs = rknn_lite.inference(inputs=[matrix1, matrix2])

# Get the result (9x9 matrix)
result = outputs[0]

# Display the result
print("Matrix 1:")
print(matrix1[0])  # Display the first (and only) matrix in the batch
print("\nMatrix 2:")
print(matrix2[0])  # Display the first (and only) matrix in the batch
np.set_printoptions(precision=3, suppress=True)
print("\nResulting matrix (Matrix 1 × Matrix 2):")
print(result[0])   # Display the first (and only) result matrix

# Verify with NumPy (optional, for correctness checking)
numpy_result = np.matmul(matrix1[0], matrix2[0])
print("\nNumPy result (for verification):")
print(numpy_result)

# Clean up
rknn_lite.release()