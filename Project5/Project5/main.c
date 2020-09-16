#include "convolution.h"
#include "args.h"
#include <stdbool.h>
#include <stdio.h>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

int main ( int argc, char *argv[] )
{
    double units;
    uint32_t size;
    float  Sigma;
    char* imgname;//the name of the image file, taken by the arguments
    int choice=0;

    //Select type of filter by input
    printf("________________________________________________\nChoose one of this filter for image convolution;\n");
    printf("\t1)Gaussian Blur\n\t2)Blur\n\t3)Sharpen\n\t4)Emboss\n\t5)Edge detection\n\t6)Sharpen5x5\n\t7)Relief\n\t8)Sobel\n");
    do{
        printf("Insert the number corresponding to the filter:");
        scanf("%d", &choice);
    }while(choice<1||choice>8);
    printf("________________________________________________\n");
    //read in the program's arguments
    if(readArguments(argc,argv,&imgname,&size,&Sigma)==false)
        return -1;

    //perform convolution on CPU 
    if(pna_kernel_cpu(imgname,size,Sigma,choice)==false)//time it
        return -2;

    //perform convolution on GPU 
    if(pna_kernel_gpu(imgname,size,Sigma,choice)==false)
		return -3;
	
    return 0;
}
