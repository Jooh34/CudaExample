
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_definitions.cuh"


__global__ void MatAdd_G2D_B2D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol);

__global__ void MatAdd_G1D_B1D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol);

__global__ void MatAdd_G2D_B1D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol);


int mainMatrixAdditionLarge(void);