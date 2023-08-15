#ifndef __MATMUL_SHARED__
#define __MATMUL_SHARED_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_definitions.cuh"

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

int mainMatmulShared(bool bUseSharedMemory);

#endif // __MATMUL_SHARED_