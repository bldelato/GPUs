#include <stdio.h>
#include "cublas.h"
#include "matrix_mul.h"
#include "cublas_v2.h"

#define DEBUG 0

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B

extern "C"
void Mul(float* A, float* B, int hA, int wA, int wB, float* C) {
	int size;
	const float alpha = 1.0f;
	const float beta = 0.0f;

	#if (DEBUG > 0)
	printf("Checkpoint! Begin Mul call (inside)\n");
	#endif

	// Load A and B to the device
	float* Ad;
	size = hA * wA * sizeof(float);
	cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	float* Bd;
	size = wA * wB * sizeof(float);
	cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, size);

	#if (DEBUG > 0)
	printf("Checkpoint! Middle Mul call (inside) Memory allocated\n");
	#endif

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t ret;

	#if (DEBUG > 0)
	printf("Checkpoint! Middle Mul call (inside) About to call cublasSgemm\n");
	#endif

	// Compute the execution configuration
	ret = cublasSgemm(
		handle,				/*cublasHandle_t handle */
		/* cublasOperation_t: CUBLAS_OP_N the non-transpose operation is selected | CUBLAS_OP_T the transpose operation is selected | CUBLAS_OP_C the conjugate transpose operation is selected */
		CUBLAS_OP_N,			/* cublasOperation_t transa */
		CUBLAS_OP_N,			/* cublasOperation_t transb */
		hA,				/* [m] */ 
		wB,				/* [n] */  
		wA,				/* [k] */ 
		&alpha,				/* alfa */ 
		Bd, wB,				/* A[m][k], num columnas (lda) */ 
		Ad, wA,				/* B[k][n], num columnas (ldb) */
		&beta,				/* beta */
		Cd, wA				/* C[m][n], num columnas (ldc) */
	);
	#if (DEBUG > 0)
	printf("Checkpoint! Middle Mul call (inside) Called cublasSgemm\n");
	#endif
	if (ret != CUBLAS_STATUS_SUCCESS) {
		printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
	}

	// Read C from the device
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);

	#if (DEBUG > 0)
	printf("Checkpoint! End Mul call (inside)\n");
	#endif
}
