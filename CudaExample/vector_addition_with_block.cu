#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer.h"

// The size of the vector
#define NUM_DATA 102490000

// Simple vector sum kernel (Max vector size : 1024)
__global__ void vecAddBlock(int* _a, int* _b, int* _c, int _size) {
	int tID = threadIdx.x;
	int bID = blockIdx.x;
	int ID = tID + bID * 1024;
	if (ID < _size)
		_c[ID] = _a[ID] + _b[ID];
}

int main(void)
{
	int* a, * b, * c, * hc;	// Vectors on the host
	int* da, * db, * dc;	// Vectors on the device

	int memSize = sizeof(int) * NUM_DATA;
	printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

	// Memory allocation on the host-side
	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);
	hc = new int[NUM_DATA]; memset(hc, 0, memSize);

	// Data generation
	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// Vector sum on host (for performance comparision)
	{
		SCOPED_TIMER("vecAdd (Host)");
		for (int i = 0; i < NUM_DATA; i++)
			hc[i] = a[i] + b[i];
	}

	// Memory allocation on the device-side
	{
		SCOPED_TIMER("cudaMalloc & cudaMemcpy Host -> Device");
		cudaMalloc(&da, memSize); cudaMemset(da, 0, memSize);
		cudaMalloc(&db, memSize); cudaMemset(db, 0, memSize);
		cudaMalloc(&dc, memSize); cudaMemset(dc, 0, memSize);

		// Data copy : Host -> Device
		cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
		cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);
	}

	// Kernel call
	{
		SCOPED_TIMER("vecAddBlock (Device)");
		vecAddBlock <<< ceil((float)NUM_DATA / 1024), 1024 >> > (da, db, dc, NUM_DATA);
		cudaDeviceSynchronize();
	}

	{
		SCOPED_TIMER("cudaMemcpy Device -> Host & cudaFree");

		// Copy results : Device -> Host
		cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);

		// Release device memory
		cudaFree(da); cudaFree(db); cudaFree(dc);
	}

	// Check results
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
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

	Timer::getInstance().printRecord();

	return 0;
}