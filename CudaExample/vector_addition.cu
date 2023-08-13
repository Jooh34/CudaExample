#include "vector_addition.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple vector sum kernel (Max vector size : 1024)
__global__ void vecAdd(int* _a, int* _b, int* _c) {
	int tID = threadIdx.x;
	_c[tID] = _a[tID] + _b[tID];
}

int mainVectorAddition(void)
{
	int* a, * b, * c, * hc;	// Vectors on the host
	int* da, * db, * dc;	// Vectors on the device

	int memSize = sizeof(int) * NUM_DATA1;
	printf("%d elements, memSize = %d bytes\n", NUM_DATA1, memSize);

	// Memory allocation on the host-side
	a = new int[NUM_DATA1]; memset(a, 0, memSize);
	b = new int[NUM_DATA1]; memset(b, 0, memSize);
	c = new int[NUM_DATA1]; memset(c, 0, memSize);
	hc = new int[NUM_DATA1]; memset(hc, 0, memSize);

	// Data generation
	for (int i = 0; i < NUM_DATA1; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// Vector sum on host (for performance comparision)
	for (int i = 0; i < NUM_DATA1; i++)
		hc[i] = a[i] + b[i];

	// Memory allocation on the device-side
	cudaMalloc(&da, memSize); cudaMemset(da, 0, memSize);
	cudaMalloc(&db, memSize); cudaMemset(db, 0, memSize);
	cudaMalloc(&dc, memSize); cudaMemset(dc, 0, memSize);

	// Data copy : Host -> Device
	cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);

	// Kernel call
	vecAdd << < 1, NUM_DATA1 >> > (da, db, dc);

	// Copy results : Device -> Host
	cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);

	// Release device memory
	cudaFree(da); cudaFree(db); cudaFree(dc);

	// Check results
	bool result = true;
	for (int i = 0; i < NUM_DATA1; i++) {
		if (hc[i] != c[i]) {
			printf("[%d] The result is not matched! (%d, %d)\n"
				, i, hc[i], c[i]);
			result = false;
		}
	}

	if (result)
		printf("GPU works well!\n");

	// Release host memory
	delete[] a; delete[] b; delete[] c;

	return 0;
}