#include "example_memory.cuh"

void checkDeviceMemory(void)
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	printf("Device memory (free/total) = %lld/%lld bytes\n", free, total);
}

int mainMemory(void)
{
	int* dDataPtr;
	cudaError_t errorCode;

	checkDeviceMemory();
	errorCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024);
	printf("cudaMalloc - %s\n", cudaGetErrorName(errorCode));
	checkDeviceMemory();

	errorCode = cudaMemset(dDataPtr, 0, sizeof(int) * 1024 * 1024);
	printf("cudaMemset - %s\n", cudaGetErrorName(errorCode));

	errorCode = cudaFree(dDataPtr);
	printf("cudaFree - %s\n", cudaGetErrorName(errorCode));
	checkDeviceMemory();

	return 0;
}