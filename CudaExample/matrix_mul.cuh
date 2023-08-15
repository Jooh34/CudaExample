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

enum BlockType {
	B1D_G2D,
	B2D_G2D,
};

int mainMatmul(BlockType blockType);

#endif // __MATMUL__