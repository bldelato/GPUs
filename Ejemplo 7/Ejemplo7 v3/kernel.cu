#include <cuda.h>
#include <math.h>

#include "kernel.h"

#include <stdio.h>



#define CUDABLOCKS 16


// Noise reduction
__global__ void noiseGPU(float *im, float *image_out, int height, int width) {

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int i = threadIdx.x, j = threadIdx.y;

	__shared__ float im_shared[CUDABLOCKS][CUDABLOCKS];
	
	if (px < 0 || px > height || py < 0 || py > width) {
		return;
	}

	im_shared[i][j] = im[px*width+py];
	__syncthreads();

	if (px < 2 || px > height-2 || py < 2 || py > width-2) {
		image_out[px*width+py] = im_shared[i][j];
		image_out[px*width+py] = 255;
		return;
        }

	if (i < 2 || i > blockDim.x-2 || j < 2 || j > blockDim.y-2 || !(i < blockDim.x-2 && j < blockDim.y-2) ) {
		// Noise reduction
		image_out[px*width+py] =
			 (2.0*im[(px-2)*width+(py-2)] +  4.0*im[(px-2)*width+(py-1)] +  5.0*im[(px-2)*width+(py)] +  4.0*im[(px-2)*width+(py+1)] + 2.0*im[(px-2)*width+(py+2)]
			+ 4.0*im[(px-1)*width+(py-2)] +  9.0*im[(px-1)*width+(py-1)] + 12.0*im[(px-1)*width+(py)] +  9.0*im[(px-1)*width+(py+1)] + 4.0*im[(px-1)*width+(py+2)]
			+ 5.0*im[(px  )*width+(py-2)] + 12.0*im[(px  )*width+(py-1)] + 15.0*im[(px  )*width+(py)] + 12.0*im[(px  )*width+(py+1)] + 5.0*im[(px  )*width+(py+2)]
			+ 4.0*im[(px+1)*width+(py-2)] +  9.0*im[(px+1)*width+(py-1)] + 12.0*im[(px+1)*width+(py)] +  9.0*im[(px+1)*width+(py+1)] + 4.0*im[(px+1)*width+(py+2)]
			+ 2.0*im[(px+2)*width+(py-2)] +  4.0*im[(px+2)*width+(py-1)] +  5.0*im[(px+2)*width+(py)] +  4.0*im[(px+2)*width+(py+1)] + 2.0*im[(px+2)*width+(py+2)])
			/159.0;
        } else {
		// Noise reduction
		image_out[px*width+py] =
			 (2.0*im_shared[(i-2)][(j-2)] +  4.0*im_shared[(i-2)][(j-1)] +  5.0*im_shared[(i-2)][(j)] +  4.0*im_shared[(i-2)][(j+1)] + 2.0*im_shared[(i-2)][(j+2)]
			+ 4.0*im_shared[(i-1)][(j-2)] +  9.0*im_shared[(i-1)][(j-1)] + 12.0*im_shared[(i-1)][(j)] +  9.0*im_shared[(i-1)][(j+1)] + 4.0*im_shared[(i-1)][(j+2)]
			+ 5.0*im_shared[(i  )][(j-2)] + 12.0*im_shared[(i  )][(j-1)] + 15.0*im_shared[(i  )][(j)] + 12.0*im_shared[(i  )][(j+1)] + 5.0*im_shared[(i  )][(j+2)]
			+ 4.0*im_shared[(i+1)][(j-2)] +  9.0*im_shared[(i+1)][(j-1)] + 12.0*im_shared[(i+1)][(j)] +  9.0*im_shared[(i+1)][(j+1)] + 4.0*im_shared[(i+1)][(j+2)]
			+ 2.0*im_shared[(i+2)][(j-2)] +  4.0*im_shared[(i+2)][(j-1)] +  5.0*im_shared[(i+2)][(j)] +  4.0*im_shared[(i+2)][(j+1)] + 2.0*im_shared[(i+2)][(j+2)])
			/159.0;
	}


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
			phi[i*width+j] = 0;
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

	//printf("phi[%i, %i]: %f\n", i, j, phi[i*width+j]);

	
}

__global__ void edgesGPU(int *pedge, float *phi, float *G, int height, int width) {

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int i = px, j = py;
		

	if (px < 3 || px > height-3 || py < 3 || py > width-3) {
		if (px >= 0 && px < height && py >= 0 && py < width) {
			//printf("[%i, %i]\n", py, px);
			pedge[i*width+j] = 0;
		}
		return;
        }

	//printf("phi[%i, %i]: %f\n", i, j, phi[i*width+j]);



	pedge[i*width+j] = 0;
	if(phi[i*width+j] == 0){
		if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
			pedge[i*width+j] = 1;

	} else if(phi[i*width+j] == 45) {
		if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
			pedge[i*width+j] = 1;

	} else if(phi[i*width+j] == 90) {
		if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
			pedge[i*width+j] = 1;

	} else if(phi[i*width+j] == 135) {
		if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
			pedge[i*width+j] = 1;
	}
}


__global__ void hysteresisThresholdingGPU(int *pedge, float *G, float *image_out, int height, int width, int level) {
	
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int i = px, j = py;
	float lowthres, hithres;

	int ii, jj;
		

	if (px < 3 || px > height-3 || py < 3 || py > width-3) {
		if (px >= 0 && px < height && py >= 0 && py < width) {
			image_out[i*width+j] = 0;
		}
		return;
        }

	lowthres = level/2;
	hithres  = 2*(level);
	image_out[i*width+j] = 0;

	if(G[i*width+j]>hithres && pedge[i*width+j])
		image_out[i*width+j] = 255;
	else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
		// check neighbours 3x3
		for (ii=-1;ii<=1; ii++)
			for (jj=-1;jj<=1; jj++)
				if (G[(i+ii)*width+j+jj]>hithres)
					image_out[i*width+j] = 255;
		
}

void cannyGPU(float *im, float *image_out, float level, int height, int width) {

	float *im_GPU, *image_out_blurred_GPU;
        float *image_out_intensitygradient_G_GPU, *image_out_intensitygradient_Gx_GPU, *image_out_intensitygradient_Gy_GPU, *image_out_intensitygradient_phi_GPU, *image_out_GPU;
	int *pedge_GPU;
	/* To fill */


	/* Mallocs GPU */
	cudaMalloc(&im_GPU, sizeof(float)*height*width);
	cudaMalloc(&image_out_blurred_GPU, sizeof(float)*height*width);

	cudaMalloc(&image_out_intensitygradient_G_GPU, sizeof(float)*height*width);
	cudaMalloc(&image_out_intensitygradient_Gx_GPU, sizeof(float)*height*width);
	cudaMalloc(&image_out_intensitygradient_Gy_GPU, sizeof(float)*height*width);
	cudaMalloc(&image_out_intensitygradient_phi_GPU, sizeof(float)*height*width);
	cudaMalloc(&pedge_GPU, sizeof(int)*height*width);
	cudaMalloc(&image_out_GPU, sizeof(float)*height*width);

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
	edgesGPU<<<dimGrid,dimBlock>>>(pedge_GPU, image_out_intensitygradient_phi_GPU, image_out_intensitygradient_G_GPU, height, width);
	cudaThreadSynchronize();
	hysteresisThresholdingGPU<<<dimGrid,dimBlock>>>(pedge_GPU, image_out_intensitygradient_G_GPU, image_out_GPU, height, width, level);
	cudaThreadSynchronize();


	/* GPU->CPU */
	cudaMemcpy(image_out, image_out_GPU, sizeof(float)*height*width, cudaMemcpyDeviceToHost);


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







