#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>

cudaError_t SelectBestGPU(int* device);
cudaError_t SelectGPU(int* device);
cudaError_t SelectGPU(int *device, const char* strDevice);
cudaError_t SelectGPU(int* device, const std::string strDevice);