#include <cuda.h>
#include <math.h>

#include "kernel.h"

#include <stdio.h>



#define CUDABLOCKS 32


// Noise reduction
__global__ void noiseGPU(float *im, float *image_out, int height, int width) {

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int i = px, j = py;
	
	

	if (px < 2 || px > height-2 || py < 2 || py > width-2) {
		if (px >= 0 && px < height && py >= 0 && py < width) {
			//printf("[%i, %i]\n", py, px);
			image_out[i*width+j] = im[i*width+j];
		}
		return;
        }

	// Noise reduction
	image_out[i*width+j] =
		 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
		+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
		+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
		+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
		+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
		/159.0;
}


// Intensity gradient of the image
__global__ void intensitygradientGPU(float *NR, float *G, float *Gx, float *Gy, float *phi, int height, int width) {
	float PI = 3.141593;

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int i = px, j = py;
		

	if (px < 2 || px > height-2 || py < 2 || py > width-2) {
		if (px >= 0 && px < height && py >= 0 && py < width) {
			//printf("[%i, %i]\n", py, px);
			phi[i*width+j] = NR[i*width+j];
		}
		return;
        }


	Gx[i*width+j] = 
		 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
		+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
		+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
		+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
		+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


	Gy[i*width+j] = 
		 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
		+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
		+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
		+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

	G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
	phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

	if(fabs(phi[i*width+j])<=PI/8 )
		phi[i*width+j] = 0;
	else if (fabs(phi[i*width+j])<= 3*(PI/8))
		phi[i*width+j] = 45;
	else if (fabs(phi[i*width+j]) <= 5*(PI/8))
		phi[i*width+j] = 90;
	else if (fabs(phi[i*width+j]) <= 7*(PI/8))
		phi[i*width+j] = 135;
	else phi[i*width+j] = 0;

	
}



void cannyGPU(float *im, float *image_out, float level, int height, int width) {

	float *im_GPU, *image_out_blurred_GPU;
        float *image_out_intensitygradient_G_GPU, *image_out_intensitygradient_Gx_GPU, *image_out_intensitygradient_Gy_GPU, *image_out_intensitygradient_phi_GPU;
	/* To fill */


	/* Mallocs GPU */
	cudaMalloc(&im_GPU, sizeof(float)*height*width);
	cudaMalloc(&image_out_blurred_GPU, sizeof(float)*height*width);

	cudaMalloc(&image_out_intensitygradient_G_GPU, sizeof(float)*height*width);
	cudaMalloc(&image_out_intensitygradient_Gx_GPU, sizeof(float)*height*width);
	cudaMalloc(&image_out_intensitygradient_Gy_GPU, sizeof(float)*height*width);
	cudaMalloc(&image_out_intensitygradient_phi_GPU, sizeof(float)*height*width);

	/* CPU->GPU */
	cudaMemcpy(im_GPU, im, sizeof(float)*height*width, cudaMemcpyHostToDevice);

	/*****************/
	/* Add Matrix GPU*/
	/*****************/
	dim3 dimBlock(CUDABLOCKS,CUDABLOCKS);
	dim3 dimGrid((height+CUDABLOCKS-1)/CUDABLOCKS, (width+CUDABLOCKS-1)/CUDABLOCKS);
	noiseGPU<<<dimGrid,dimBlock>>>(im_GPU, image_out_blurred_GPU, height, width);
	cudaThreadSynchronize();
	intensitygradientGPU<<<dimGrid,dimBlock>>>(image_out_blurred_GPU, image_out_intensitygradient_G_GPU, image_out_intensitygradient_Gx_GPU, image_out_intensitygradient_Gy_GPU, image_out_intensitygradient_phi_GPU, height, width);
	cudaThreadSynchronize();

	/* GPU->CPU */
	cudaMemcpy(image_out, image_out_intensitygradient_phi_GPU, sizeof(float)*height*width, cudaMemcpyDeviceToHost);


	cudaFree(image_out_blurred_GPU);
	cudaFree(image_out_intensitygradient_G_GPU);
	cudaFree(image_out_intensitygradient_Gx_GPU);
	cudaFree(image_out_intensitygradient_Gy_GPU);
	cudaFree(image_out_intensitygradient_phi_GPU);
}






/*
        int i, j;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			image_out[i*width+j] = im[i*width+j];	
		}
	}

*/







