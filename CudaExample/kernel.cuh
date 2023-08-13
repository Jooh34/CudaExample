#ifndef __KERNEL__
#define __KERNEL__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addKernel(int* c, const int* a, const int* b);

int mainKernel();

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

#endif // __KERNEL__