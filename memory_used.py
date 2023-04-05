import psutil
import GPUtil

# Set the threshold for GPU memory usage (in MB)
threshold = 2000

# Get a list of all GPUs
gpus = GPUtil.getGPUs()

# Loop over each GPU and check its memory usage
for gpu in gpus:
    # Get the GPU memory usage in MB
    mem_used = gpu.memoryUsed

    # If memory usage is above the threshold, print a message
    #if mem_used > threshold:
    print(f"GPU {gpu.id} memory usage is {mem_used} MB")
