#ifndef __VECTOR_ADDITION_LARGE__
#define __VECTOR_ADDITION_LARGE__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The size of the vector
#define NUM_DATA 123456789

// Simple vector sum kernel (Max vector size : 1024)
__global__ void vecAddLarge(int* _a, int* _b, int* _c, int _size);

int mainVectorAdditionLarge(void);

#endif // __VECTOR_ADDITION_LARGE__