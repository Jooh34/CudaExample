
#include "matrix_addition_large.cuh"

#include "timer.h"

#define ROW_SIZE (8192)
#define COL_SIZE (8192)
#define MAT_SIZE (ROW_SIZE*COL_SIZE)

#define ID2INDEX(_row,_col) (_row*COL_SIZE + _col)

#define NUM_LAYOUTS 3

#define G2D_B2D 0
#define G1D_B1D 1
#define G2D_B1D 2


__global__ void MatAdd_G2D_B2D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol)
{
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int ID = nCol * col + row;
	if (col < nCol && row < nRow)
		MatC[ID] = MatA[ID] + MatB[ID];
}

__global__ void MatAdd_G1D_B1D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol)
{
	unsigned int col = blockDim.x * threadIdx.x + blockIdx.x;

	if (col < nCol) {
		for (int row = 0; row < nRow; row++) {
			int index = row * COL_SIZE + col;
			MatC[index] = MatA[index] + MatB[index];
		}
	}
}

__global__ void MatAdd_G2D_B1D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol)
{
	// Write your 2D_1D kernel here

	unsigned int col = blockDim.x * threadIdx.x + blockIdx.x;
	unsigned int row = blockIdx.y;
	unsigned int ID = col * nCol + row;
	if (col < nCol && row < nRow) {
		MatC[ID] = MatA[ID] + MatB[ID];
	}
}

int mainMatrixAdditionLarge(void)
{

	float* A, * B, * C[NUM_LAYOUTS], * hC;
	float* dA, * dB, * dC;

	//// host memory allocation
	allocNinitMem<float>(&A, MAT_SIZE);
	allocNinitMem<float>(&B, MAT_SIZE);
	for (int i=0; i< NUM_LAYOUTS; i++) {
		allocNinitMem<float>(&C[i], MAT_SIZE);
	}

	allocNinitMem<float>(&hC, MAT_SIZE);

	// device memory allocation
	cudaMalloc(&dA, sizeof(float) * MAT_SIZE); cudaMemset(dA, 0, sizeof(float) * MAT_SIZE);
	cudaMalloc(&dB, sizeof(float) * MAT_SIZE); cudaMemset(dB, 0, sizeof(float) * MAT_SIZE);
	cudaMalloc(&dC, sizeof(float) * MAT_SIZE); cudaMemset(dC, 0, sizeof(float) * MAT_SIZE);

	// input matrix generation
	for (int i = 0; i < MAT_SIZE; i++) {
		A[i] = rand() % 100;
		B[i] = rand() % 100;
	}

	{
		SCOPED_TIMER("Host Performance")
		for (int row = 0; row < ROW_SIZE; row++) {
			for (int col = 0; col < COL_SIZE; col++) {
				hC[ID2INDEX(row, col)] = A[ID2INDEX(row, col)] + B[ID2INDEX(row, col)];
			}
		}
	}


	// copy the input matrices from host memory to device memory
	{
		SCOPED_TIMER("Data Trans. : Host -> Device");
		cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE, cudaMemcpyHostToDevice);
	}

	// *** Kernel call
	{
		SCOPED_TIMER("G2D_B2D");
		dim3 blockDim(32, 32, 1);
		dim3 gridDim(ceil((float)COL_SIZE / blockDim.x), ceil((float)ROW_SIZE / blockDim.y), 1);
		MatAdd_G2D_B2D<<<gridDim, blockDim>>>(dA, dB, dC, ROW_SIZE, COL_SIZE);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(C[G2D_B2D], dC, sizeof(float) * MAT_SIZE, cudaMemcpyDeviceToHost);

	{
		SCOPED_TIMER("G1D_B1D");
		dim3 blockDim(32, 1, 1);
		dim3 gridDim(ceil((float)COL_SIZE / blockDim.x), 1, 1);
		MatAdd_G1D_B1D << <gridDim, blockDim >> > (dA, dB, dC, ROW_SIZE, COL_SIZE);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(C[G1D_B1D], dC, sizeof(float) * MAT_SIZE, cudaMemcpyDeviceToHost);

	{
		SCOPED_TIMER("G2D_B1D");
		dim3 blockDim(32, 1, 1);
		dim3 gridDim(ceil((float)COL_SIZE / blockDim.x), ROW_SIZE, 1);
		MatAdd_G2D_B1D << <gridDim, blockDim >> > (dA, dB, dC, ROW_SIZE, COL_SIZE);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(C[G2D_B1D], dC, sizeof(float) * MAT_SIZE, cudaMemcpyDeviceToHost);

	// ***

	// validation
	bool isCorrect = true;
	for (int layout = 0; layout < NUM_LAYOUTS; layout++) {
		isCorrect = true;
		for (int i = 0; i < MAT_SIZE; i++) {
			if (hC[i] != C[layout][i]) {
				isCorrect = false;
				break;
			}
		}

		switch (layout) {
		case G1D_B1D:
			printf("G1D_B1D");
			break;
		case G2D_B1D:
			printf("G2D_B1D");
			break;
		case G2D_B2D:
			printf("G2D_B2D");
			break;
		}
		if (isCorrect) printf(" kernel works well!\n");
		else printf(" kernel fails to make correct result(s)..\n");
	}

	Timer::getInstance().printRecord();

	SAFE_DELETE(A);
	SAFE_DELETE(B);
	SAFE_DELETE(hC);
	for (int i=0; i<NUM_LAYOUTS; i++) { SAFE_DELETE(C[i]); }
	return 0;
}