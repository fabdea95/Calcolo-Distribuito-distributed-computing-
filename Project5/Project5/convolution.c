#include "bitmap.h"
#include "convolution.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define PI_ 3.14159265359f
#define MAX_SOURCE_SIZE (1048576) //1 MB
#define MAX_LOG_SIZE    (1048576) //1 MB
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
float* matrix;
float* value;

//creates a gaussian kernel (Choice 1)
float* createGaussianKernel(uint32_t size,float sigma)
{
    float* ret;
    uint32_t x,y;
    double center = size/2;
    float sum = 0;
    //allocate and create the gaussian kernel
    ret = malloc(sizeof(float) * size * size);
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            ret[ y*size+x] = exp( (((x-center)*(x-center)+(y-center)*(y-center))/(2.0f*sigma*sigma))*-1.0f ) / (2.0f*PI_*sigma*sigma);
            sum+=ret[ y*size+x];
        }
    }
    //normalize
    for(x = 0; x < size*size;x++)
    {
        ret[x] = ret[x]/sum;
    }
    //print the kernel so the user can see it
    printf("The generated Gaussian Kernel is:\n");
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            printf("%f ",ret[ y*size+x]);
        }
        printf("\n");
    }
    return ret;
}

//creates a edge detection kernel (Choice 5)
float* createEdgeDetKernel(uint32_t size)//size filter
{
    float* ret;
    uint32_t x,y;
    double center = size/2;
    float sum = 0;
    int filter[9]= {-1,-1,-1,
                    -1, 8,-1,
                    -1,-1,-1};
    //allocate and create the kernel
    ret = malloc(sizeof(float) * 3 * 3);
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            ret[ y*size+x] = filter[y*size+x];
        }
    }
    //print the kernel so the user can see it
    printf("The generated Edge Detection Kernel is:\n");
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            printf(" %i ",(int)ret[ y*size+x]);
        }
        printf("\n");
    }
    return ret;
}

//creates a Sharpen kernel (Choice 3)
float* createSharpen(uint32_t size)//size filter
{
    float* ret;
    uint32_t x,y;
    double center = size/2;
    float sum = 0;
    float filter[9]= {0.0,-1.0,0.0,
                    -1.0,5.0,-1.0,
                    0.0,-1.0,0.0};
    //allocate and create the kernel
    ret = malloc(sizeof(float) * 3 * 3);
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            ret[ y*size+x] = filter[y*size+x];
        }
    }
    //print the kernel so the user can see it
    printf("The generated Sharpen Kernel is:\n");
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            printf(" %d ",(int)ret[ y*size+x]);
        }
        printf("\n");
    }
	//system("pause");
    return ret;
}

//creates a emboss kernel (Choice 4)
float* createEmboss(uint32_t size)//size filter
{
    float* ret;
    uint32_t x,y;
    double center = size/2;
    float sum = 0;
    float filter[9]= {-2.0,-1.0,0.0,
                    -1.0,1.0,1.0,
                    0.0,1.0,2.0};
	printf("%d\n", size);
    //allocate and create the kernel
    ret = malloc(sizeof(float) * 3 * 3);
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            ret[ y*size+x] = filter[y*size+x];
        }
    }
    //print the kernel so the user can see it
    printf("The generated Emboss Kernel is:\n");
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            printf(" %i ",(int)ret[ y*size+x]);
        }
        printf("\n");
    }
    return ret;
}

//creates a Blur kernel (Choice 2)
float* createBlur(uint32_t size)//size filter
{
    float* ret;
    uint32_t x,y;
    double center = size/2;
    float sum = 0;
	printf("%d\n",size);
    float filter[9]= {0.0, 0.25, 0.0,
                    0.25, 0.0, 0.25,
                    0.0, 0.25, 0.0};
    //allocate and create the kernel
    ret = malloc(sizeof(float) * 3 * 3);
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            ret[ y*size+x] = filter[y*size+x];
        }
    }
    //print the kernel so the user can see it
    printf("The generated Blur Kernel is:\n");
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            printf(" %f ",ret[ y*size+x]);
        }
        printf("\n");
    }
    return ret;
}
//creates a Sharpen5x5 kernel (Choice 6)
float* createSharpen5(uint32_t size)//size filter
{
    float* ret;
    uint32_t x,y;
    double center = 5/2;
    float sum = 0;
    int filter[25]= {-1, -3, -4, -3, -1, 
                    -3,  0,  6,  0, -3, 
                    -4,  6, 21,  6, -4, 
                    -3,  0,  6,  0, -3,
                    -1, -3, -4, -3, -1};
    //allocate and create the kernel
    ret = malloc(sizeof(float) * 5 * 5);
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            ret[ y*size+x] = filter[y*size+x];
        }
    }
    //print the kernel so the user can see it
    printf("The generated Sharpen Kernel is:\n");
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            printf(" %i ",(int)ret[ y*size+x]);
        }
        printf("\n");
    }
    return ret;
}

//creates a Relief kernel (Choice 7)
float* createRelief(uint32_t size)//size filter
{
    float* ret;
    uint32_t x,y;
    double center = size/2;
    float sum = 0;
	
    int filter[9]= {2,1,0,
                    1,1,-1,
                    0,-1,-2};
    //allocate and create the kernel
    ret = malloc(sizeof(float) * 3 * 3);
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            ret[ y*size+x] = filter[y*size+x];
        }
    }
    //print the kernel so the user can see it
    printf("The generated Relief Kernel is:\n");
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            printf(" %d ",(int)ret[ y*size+x]);
        }
        printf("\n");
    }
    return ret;
}
//creates a Sobel kernel (Choice 8)
float* createSobel(uint32_t size)//size filter
{
    float* ret;
    uint32_t x,y;
    double center = size/2;
    float sum = 0;
    int filter[9]= {1,0,-1,
                    2,0,-2,
                    1,0,-1};
    //allocate and create the kernel
    ret = malloc(sizeof(float) * 3 * 3);
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            ret[ y*size+x] =filter[y*size+x];
        }
    }
    //print the kernel so the user can see it
    printf("The generated Sobel Kernel is:\n");
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            printf(" %d ",(int) ret[ y*size+x]);
        }
        printf("\n");
    }
    return ret;
}


//given image using the CPU algorithm
char pna_kernel_cpu(char* imgname,uint32_t size,float sigma, int choise)
{
	
    uint32_t i,x,y,imgLineSize;
    int32_t center,yOff,xOff;
	//float* matrix;
	float value;
    switch(choise){
	case 1: {
		matrix = createGaussianKernel(size, sigma);
		break;
	}
	case 5: {
		matrix = createEdgeDetKernel(3);
		size = 3;
		break;
	}
	case 3: {
		matrix = createSharpen(3);
		size = 3;
		break;
	}
	case 4: {
		matrix = createEmboss(3);
		size = 3;
		break;
	}
	case 2: {
		matrix = createBlur(3);
		size = 3;
		break;
	}
	case 6: {
		matrix = createSharpen5(5);
		size = 5;
		break;
	}
	case 7: {
		matrix = createRelief(3);
		size = 3;
		break;
	}
	case 8: {
		matrix = createSobel(3);
		size = 3;
		break;
	}
    }
//	int* matrix;
    //begin time
    clock_t begin=clock();
    //read the bitmap
    ME_ImageBMP bmp;
    if(meImageBMP_Init(&bmp,imgname)==false)
    {
        printf("Image \"%s\" could not be read as a .BMP file\n",imgname);
        return false;
    }
    //find the size of one line of the image in bytes and the center of the kernel
    imgLineSize = bmp.imgWidth*3;
    center = size/2;
	printf("pna size:   %d\n", size);
	//convolve all valid pixels with the kernel
	char* buf;
	char appoggio[10];
	int imgSize = bmp.imgWidth*bmp.imgHeight * 3;
	buf = malloc(imgSize);
	//appoggio = malloc(7);
	for (i = imgLineSize * (size - center) + center * 3; i < bmp.imgHeight*bmp.imgWidth * 3 - imgLineSize * (size - center) - center * 3; i++)
	{
		
		value = 0;
		for (y = 0; y < size; y++)
		{
			yOff = imgLineSize * (y - center);
			for (x = 0; x < size; x++)
			{
				xOff = 3 * (x - center);
				value += matrix[y*size + x] * bmp.imgData[i + xOff + yOff];
				//printf("%s \t", bmp.imgData[i + xOff + yOff]);
			}
		}
		_gcvt(value, 6, appoggio);
		//sprintf(appoggio, "%f", value);
		//strncat(buf, appoggio, 6);



		//printf("%s \t", buf);
		//strncat(bmp.imgData, buf, 6);
		
		//printf("value: %f\n", value);
		//printf("%s \n", buf);
		
		//strcpy(buf, appoggio);
		//free(appoggio);
		//bmp.imgData[i] = buf[i];

		//old version: 			bmp.imgData[i] = value;
		// new version:
		
		//printf("value: %f\n", value);
		//printf("imgData: %s", bmp.imgData);
    }
    //free memory and save the image
    free(matrix);
	bmp.imgData= buf;
	free(buf);
	//free(appoggio);
	clock_t end = clock();
	//print the time elapsed (=end-begin)
	printf("Time taken for convolution with CPU:%lf\n\n", (double)(end - begin) / CLOCKS_PER_SEC);
    meImageBMP_Save(&bmp,"cpu.bmp");
    //end time
  
	system("pause");
    return true;
}

//given image using the GPU
char pna_kernel_gpu(char* imgname,uint32_t size,float sigma, int choise)
{
    uint32_t imgSize;
    //float* matrix;
    cl_int ret;//the openCL error code
	switch (choise) {
	case 1: {
		matrix = createGaussianKernel(size, sigma);
		break;
	}
	case 5: {
		matrix = createEdgeDetKernel(3);
		size = 3;
		break;
	}
	case 3: {
		matrix = createSharpen(3);
		size = 3;
		break;
	}
	case 4: {
		matrix = createEmboss(3);
		size = 3;
		break;
	}
	case 2: {
		matrix = createBlur(3);
		size = 3;
		break;
	}
	case 6: {
		matrix = createSharpen5(5);
		size = 5;
		break;
	}
	case 7: {
		matrix = createRelief(3);
		size = 3;
		break;
	}
	case 8: {
		matrix = createSobel(3);
		size = 3;
		break;
	}
	}
	
    //begin time
    clock_t start=clock();
    //get the image
    ME_ImageBMP bmp;
    meImageBMP_Init(&bmp,imgname);
    imgSize = bmp.imgWidth*bmp.imgHeight*3;
    //create the pointer that will hold the new (blurred) image data
    unsigned char* newData;
    newData = malloc(imgSize);
    // Read in the kernel code into a c string
    FILE* f;
    char* kernelSource;
    size_t kernelSrcSize;
   if( (f = fopen("kernel.cl", "r")) == NULL)
    {
        fprintf(stderr, "Failed to load OpenCL kernel code.\n");
        return false;
    }
    kernelSource = malloc(MAX_SOURCE_SIZE);
    kernelSrcSize = fread( kernelSource, 1, MAX_SOURCE_SIZE, f);
    fclose(f);

    //Get platform and device information
    cl_platform_id platformID;//will hold the ID of the openCL available platform
    cl_uint platformsN;//will hold the number of openCL available platforms on the machine
    cl_device_id deviceID;//will hold the ID of the openCL device
    cl_uint devicesN; //will hold the number of OpenCL devices in the system
    if(clGetPlatformIDs(1, &platformID, &platformsN) != CL_SUCCESS)
    {
        printf("Could not get the OpenCL Platform IDs\n");
        return false;
    }
    if(clGetDeviceIDs(platformID, CL_DEVICE_TYPE_DEFAULT, 1,&deviceID, &devicesN) != CL_SUCCESS)
    {
        printf("Could not get the system's OpenCL device\n");
        return false;
    }
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &deviceID, NULL, NULL, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create a valid OpenCL context\n");
        return false;
    }
    // Create a command queue
    cl_command_queue cmdQueue = clCreateCommandQueue(context, deviceID, 0, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create an OpenCL Command Queue\n");
        return false;
    }

    // Create memory buffers on the device for the two images
    cl_mem gpuImg = clCreateBuffer(context,CL_MEM_READ_ONLY,imgSize,NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the GPU image buffer object\n");
        return false;
    }
    cl_mem gpuFilter = clCreateBuffer(context,CL_MEM_READ_ONLY,size*size*sizeof(float),NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the GPU image buffer object\n");
        return false;
    }
    cl_mem gpuNewImg = clCreateBuffer(context,CL_MEM_WRITE_ONLY,imgSize,NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the GPU image buffer object\n");
        return false;
    }
    //Copy the image data and the kernel to the memory buffer
    if(clEnqueueWriteBuffer(cmdQueue, gpuImg, CL_TRUE, 0,imgSize,bmp.imgData, 0, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error during sending the image data to the OpenCL buffer\n");
        return false;
    }
    if(clEnqueueWriteBuffer(cmdQueue, gpuFilter, CL_TRUE, 0,size*size*sizeof(float),matrix, 0, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error during sending the kernel to the OpenCL buffer\n");
        return false;
    }
    //Create a program object and associate it with the kernel's source code.
    cl_program program = clCreateProgramWithSource(context, 1,(const char **)&kernelSource, (const size_t *)&kernelSrcSize, &ret);
    free(kernelSource);
    if(ret != CL_SUCCESS)
    {
        printf("Error in creating an OpenCL program object\n");
        return false;
    }
    //Build the created OpenCL program
    if((ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL))!= CL_SUCCESS)
    {
        printf("Failed to build the OpenCL program\n");
        //create the log string and show it to the user. Then quit
        char* buildLog;
        buildLog = malloc(MAX_LOG_SIZE);
        if(clGetProgramBuildInfo(program,deviceID,CL_PROGRAM_BUILD_LOG,MAX_LOG_SIZE,buildLog,NULL) != CL_SUCCESS)
        {
            printf("Could not get any Build info from OpenCL\n");
            free(buildLog);
            return false;
        }
        printf("**BUILD LOG**\n%s",buildLog);
        free(buildLog);
        return false;
    }
    // Create the OpenCL kernel. This is basically one function of the program declared with the __kernel qualifier
    cl_kernel kernel = clCreateKernel(program, "convolution_kernel", &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Failed to create the OpenCL Kernel from the built program\n");
        return false;
    }
    ///Set the arguments of the kernel
    if(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gpuImg) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"gpuImg\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gpuFilter) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"gpuFilter\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 2, sizeof(int), (void *)&bmp.imgWidth) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"imageWidth\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 3, sizeof(int), (void *)&bmp.imgHeight) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"imgHeight\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel,4,sizeof(int),(void*)&size) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"size\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&gpuNewImg) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"gpuNewImg\" argument\n");
        return false;
    }

    ///enqueue the kernel into the OpenCL device for execution
    size_t globalWorkItemSize = imgSize;//the total size of 1 dimension of the work items. Basically the whole image buffer size
    size_t workGroupSize = 64; //The size of one work group
    ret = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, &globalWorkItemSize, &workGroupSize,0, NULL, NULL);


    ///Read the memory buffer of the new image on the device to the new Data local variable
    ret = clEnqueueReadBuffer(cmdQueue, gpuNewImg, CL_TRUE, 0,imgSize, newData, 0, NULL, NULL);

    ///Clean up everything
    free(matrix);
	clock_t End = clock();
	printf("Time taken for convolution with GPU:%lf\n", (double)(End - start) / CLOCKS_PER_SEC);
    clFlush(cmdQueue);
    clFinish(cmdQueue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(gpuImg);
    clReleaseMemObject(gpuFilter);
    clReleaseMemObject(gpuNewImg);
    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);
    ///save the new image and return success
    bmp.imgData = newData;
    meImageBMP_Save(&bmp,"gpu.bmp");
    
	system("pause");
	return true;
}
