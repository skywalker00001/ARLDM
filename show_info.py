import torch
import platform

print(f"PyTorch version: {torch.__version__}")
print(f"Is debug build: {torch.version.debug}")
print(f"CUDA used to build PyTorch: {torch.version.cuda}")
print(f"ROCM used to build PyTorch: {torch.version.hip}")
print(f"OS: {platform.platform()}")
# print(f"GCC version: {torch._C._gcc_version()}")
# print(f"Clang version: {torch._C._clang_version()}")
# print(f"CMake version: {torch._C._cmake_version()}")
# print(f"Python version: {torch._C.python_version()}")
# print(f"Is CUDA available: {torch.cuda.is_available()}")
#print(f"CUDA runtime version: {torch.version.cuda_runtime_version}")
print(f"GPU models and configuration: {torch.cuda.get_device_name()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

#print(f"Nvidia driver version: {torch._C._cuda_driver_version()}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
#print(f"HIP runtime version: {torch.version.hip_runtime_version}")
#print(f"MIOpen runtime version: {torch.version.mio_version()}")

print("Versions of relevant libraries:")
#print(f"{torch._C._show_version()}")