__kernel void matrixMul(
	__global int *A,
	__global int *B,
	__global int *C,
	int n, int m) {
	int tx = get_global_id(0);
	int ty = get_global_id(1);
	int value = 0;
	for (int k = 0; k < n; ++k)
	{
		int elementA = A[ty * n + k];
		int elementB = B[k * m + tx];
		value += elementA * elementB;
	}
	// Scrive la matrice nella device memory; ogni thread scrive un elemento
	C[ty * n + tx] = value;
}