#include "matrix_mul_large_shared.cuh"

// Matrix size
#define SIZE_M (512*2)
#define SIZE_N (512*4)
#define SIZE_K (512*2)

#define DATA_TYPE int

#define BLOCK_X 4
#define BLOCK_Y 4

__global__ void MatMul_xRow(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int M, int N, int K)
{
	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

	DATA_TYPE result = 0;
	if (row < M && col < N) {
		for (int k = 0; k < K; k++) {
			result += (matA[row * K + k] * matB[k * N + col]);
		}
	}
	matC[row * N + col] = result;
}

__global__ void MatMul_yRow(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int M, int N, int K)
{
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

	DATA_TYPE result = 0;
	if (row < M && col < N) {
		for (int k = 0; k < K; k++) {
			result += (matA[row * K + k] * matB[k * N + col]);
		}
	}
	matC[row * N + col] = result;
}

int mainMatmulMemaccOpt(bool bYRow)
{
	// set matrix size
	int m, n, k;
	m = SIZE_M;
	n = SIZE_N;
	k = SIZE_K;

	printf("Size : A = (%d by %d), B = (%d by %d), C = (%d by %d)\n", m, k, k, n, m, n);

	int sizeA = m * k;
	int sizeB = k * n;
	int sizeC = m * n;

	// Make matrix
	DATA_TYPE* A = NULL, * B = NULL;
	allocNinitMem<DATA_TYPE>(&A, sizeA);
	allocNinitMem<DATA_TYPE>(&B, sizeB);

	DATA_TYPE* Ccpu = NULL, * Cgpu = NULL;
	allocNinitMem<DATA_TYPE>(&Ccpu, sizeC);
	allocNinitMem<DATA_TYPE>(&Cgpu, sizeC);

	// generate input matrices
	for (int i = 0; i < sizeA; i++) A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
	for (int i = 0; i < sizeB; i++) B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));

	{
		SCOPED_TIMER("CPU matmul");
		// CPU matmul
		for (int row = 0; row < m; row++) {
			for (int col = 0; col < n; col++) {
				int cIndex = row * n + col;
				Ccpu[cIndex] = 0;
				for (int i = 0; i < k; i++)
					Ccpu[cIndex] += (A[row * k + i] * B[i * n + col]);
			}
		}
	}

	/******************************************************************
	* Write your codes for GPU algorithm from here
	******************************************************************/
	DATA_TYPE* dA, * dB, * dC;

	// 1. Allocate device memory for dA, dB, dC
	{
		SCOPED_TIMER("CUDA malloc and memset");
		cudaMalloc(&dA, sizeof(DATA_TYPE) * sizeA); cudaMemset(dA, 0, sizeof(DATA_TYPE) * sizeA);
		cudaMalloc(&dB, sizeof(DATA_TYPE) * sizeB); cudaMemset(dB, 0, sizeof(DATA_TYPE) * sizeB);
		cudaMalloc(&dC, sizeof(DATA_TYPE) * sizeC); cudaMemset(dC, 0, sizeof(DATA_TYPE) * sizeC);
	}

	// 2. Send(Copy) the input matrices to GPU (A -> dB, B -> dB)
	{
		SCOPED_TIMER("CUDA copy Host -> Device");
		cudaMemcpy(dA, A, sizeof(DATA_TYPE) * sizeA, cudaMemcpyHostToDevice);
		cudaMemcpy(dB, B, sizeof(DATA_TYPE) * sizeB, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}


	// 3. Set the thread layout
	if (bYRow)
	{
		dim3 blockDim(BLOCK_X, BLOCK_Y, 1);
		dim3 gridDim(ceil(float(n) / blockDim.x), ceil(float(m) / blockDim.y), 1);

		printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

		// 4. Kernel call
		{
			SCOPED_TIMER("Matmul on GPU");
			MatMul_yRow << < gridDim, blockDim >> > (dA, dB, dC, m, n, k);
			cudaDeviceSynchronize(); // this is synchronization for mearusing the kernel processing time
		}
	}
	else
	{
		dim3 blockDim(BLOCK_X, BLOCK_Y, 1);
		dim3 gridDim(ceil(float(m) / blockDim.x), ceil(float(n) / blockDim.y), 1);

		printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

		// 4. Kernel call
		{
			SCOPED_TIMER("Matmul on GPU");
			MatMul_xRow << < gridDim, blockDim >> > (dA, dB, dC, m, n, k);
			cudaDeviceSynchronize(); // this is synchronization for mearusing the kernel processing time
		}

	}

	//5. Get(copy) the result from GPU to host memory (dC -> Cgpu)
	{
		SCOPED_TIMER("CUDA copy Device -> Host");
		cudaMemcpy(Cgpu, dC, sizeof(DATA_TYPE) * sizeC, cudaMemcpyDeviceToHost);
	}

	// 6. Release device memory space (dA, dB, dC)
	cudaFree(dA); cudaFree(dB); cudaFree(dC);


	compareMatrix<DATA_TYPE>(Ccpu, Cgpu, sizeC);

	Timer::getInstance().printRecord();

	delete A;
	delete B;
	delete Ccpu;
	delete Cgpu;

	return 0;
}
