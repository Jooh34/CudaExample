#ifndef __VECTOR_ADDITION__
#define __VECTOR_ADDITION__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The size of the vector
#define NUM_DATA1 1024

// Simple vector sum kernel (Max vector size : 1024)
__global__ void vecAdd(int* _a, int* _b, int* _c);

int mainVectorAddition(void);

#endif // __VECTOR_ADDITION__