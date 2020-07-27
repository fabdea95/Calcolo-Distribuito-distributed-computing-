#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define _CRT_SECURE_NO_WARNINGS

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
	// elementi per lato:
	/*const int rowsA = 2500;
	const int colsA = 2500;
	const int rowsB = 2500;
	const int colsB = 2500;*/
	const int rows = 2500;
	const int cols = 2500;


	if (cols != rows) {
		printf("Error: dimensioni matrici incompatibili");
		return(1);
	}

	const char* source_str = "__kernel void matrixMul(__global int *C, __global int *A, __global int *B, int n, int m) { int tx = get_global_id(0); int ty = get_global_id(1); int value = 0; for (int k = 0; k < n; ++k) { int elementA = A[ty * n + k];		int elementB = B[k * m + tx];			value += elementA * elementB; }			C[ty * n + tx] = value;}";
	size_t source_size;

	long int mem_size_A = sizeof(int) * (rows*cols);;
	int* A = (int*)malloc(mem_size_A);
	

	long int mem_size_B = sizeof( int) * (rows*cols);;
	int* B = ( int*)malloc(mem_size_B);

	long int mem_size_C = sizeof(int) * (rows*cols);;
	int* C = (int*)malloc(mem_size_B);
	//RIEMPIMENTO MATRICI:
	for (int i = 0; i < (rows*cols); i++) {
	    A[i] = i;
		B[i] = 1;
		C[i] = 0;
	}
	
	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	// Connessione device
	int gpu = 1;
	cl_device_id device_id;
	cl_int err;
	err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1,
		&device_id, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: creazione devices fallita!\n");
		return EXIT_FAILURE;
	}

	// Creazione contesto
	cl_context context = NULL;
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: creazione contesto fallita!\n");
		return EXIT_FAILURE;
	}

	// Creazione command queue
	cl_command_queue commands;
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: creazione command queue fallita!\n");
		return EXIT_FAILURE;
	}
	
	cl_program program = NULL;
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &err);
	if (!program)
	{
		printf("Error: creazione programma fallita!\n");
		return EXIT_FAILURE;
	}
	
	// Build dell`eseguibile
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		printf("Error: build fallito!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
			buffer, &len);

		printf("%s\n", buffer);
		system("pause");
		exit(1);
	}
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "matrixMul", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: creazione kernel fallita!\n");
		exit(1);
	}
	cl_mem b_A = NULL;
	cl_mem b_B = NULL;
	cl_mem b_C = NULL;
	b_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
	b_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, A,
		&err);
	b_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, B,
		&err);

	if (!b_A || !b_B || !b_C)
	{
		printf("Error: allocazione memoria fallita!\n");
		exit(1);
	}

	printf("Running matrix multiplication for matrices A (%dx%d) and B (%dx%d) ...\n",
		rows, cols, rows, cols);

	//Launch OpenCL kernel
	size_t localWorkSize[2], globalWorkSize[2];

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&b_C);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_A);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&b_B);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	printf("OK");
	if (err != CL_SUCCESS)
	{
		printf("Error: passaggio argomenti al kernel fallito! %d\n", err);
		system("pause");
		exit(1);
	}

	localWorkSize[0] = 16;
	localWorkSize[1] = 16;
	globalWorkSize[0] = 1024;
	globalWorkSize[1] = 1024;

	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize,
		0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: esecuzione kernel fallita! %d\n", err);
		system("pause");
		exit(1);
	}

	//Retrieve result from device
	err = clEnqueueReadBuffer(commands, b_C, CL_TRUE, 0, mem_size_C, C, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: lettura array in input fallita! %d\n", err);
		exit(1);
	}
	printf("Matrix multiplication completata...\n");
	printf("\n\nMatrix C (Results)\n");
	int i;
	/*
	STAMPA MATRICE RISULTATO
	for (i = 0; i < (rows*cols); i++)
	{
		printf("%d ", C[i]);
		if (((i + 1) % cols) == 0)
			printf("\n");
	}
	printf("\n");
	*/

	//Shutdown and cleanup
	free(A);
	free(B);
	free(C);

	clReleaseMemObject(b_A);
	clReleaseMemObject(b_C);
	clReleaseMemObject(b_B);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	//system("pause");
	return 0;
	
}