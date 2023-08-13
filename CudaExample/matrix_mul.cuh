#ifndef __MATMUL__
#define __MATMUL__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_definitions.cuh"

#define DO_CPU
#define DATA_TYPE int

enum BlockType {
	B1D_G2D,
	B2D_G2D,
};

__global__ void MatMul(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int M, int N, int K);
int mainMatmul(BlockType blockType);
bool compareMatrix(DATA_TYPE* _A, DATA_TYPE* _B, int _size);

#endif // __MATMUL__