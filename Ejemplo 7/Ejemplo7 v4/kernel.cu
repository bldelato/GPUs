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

	if (i < 2 || i >= blockDim.x-2 || j < 2 || j >= blockDim.y-2 ) {
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
__global__ void intensitygradientGPU(float *NR, float *G, float *phi, int height, int width) {
	float PI = 3.141593;

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int i = threadIdx.x, j = threadIdx.y;
	int imagePixel = px*width+py;
	float Gx, Gy;
	
	__shared__ float NR_shared[CUDABLOCKS][CUDABLOCKS];

	if (px < 0 || px > height || py < 0 || py > width) {
		return;
	}
	NR_shared[i][j] = NR[imagePixel];
	__syncthreads();

	if (px < 2 || px > height-2 || py < 2 || py > width-2) {
		phi[px*width+py] = 0;
		return;
        }

	if (i < 2 || i >= blockDim.x-2 || j < 2 || j >= blockDim.y-2 ) {
		Gx = 
			 (1.0*NR[(px-2)*width+(py-2)] +  2.0*NR[(px-2)*width+(py-1)] +  (-2.0)*NR[(px-2)*width+(py+1)] + (-1.0)*NR[(px-2)*width+(py+2)]
			+ 4.0*NR[(px-1)*width+(py-2)] +  8.0*NR[(px-1)*width+(py-1)] +  (-8.0)*NR[(px-1)*width+(py+1)] + (-4.0)*NR[(px-1)*width+(py+2)]
			+ 6.0*NR[(px  )*width+(py-2)] + 12.0*NR[(px  )*width+(py-1)] + (-12.0)*NR[(px  )*width+(py+1)] + (-6.0)*NR[(px  )*width+(py+2)]
			+ 4.0*NR[(px+1)*width+(py-2)] +  8.0*NR[(px+1)*width+(py-1)] +  (-8.0)*NR[(px+1)*width+(py+1)] + (-4.0)*NR[(px+1)*width+(py+2)]
			+ 1.0*NR[(px+2)*width+(py-2)] +  2.0*NR[(px+2)*width+(py-1)] +  (-2.0)*NR[(px+2)*width+(py+1)] + (-1.0)*NR[(px+2)*width+(py+2)]);

		Gy = 
			 ((-1.0)*NR[(px-2)*width+(py-2)] + (-4.0)*NR[(px-2)*width+(py-1)] +  (-6.0)*NR[(px-2)*width+(py)] + (-4.0)*NR[(px-2)*width+(py+1)] + (-1.0)*NR[(px-2)*width+(py+2)]
			+ (-2.0)*NR[(px-1)*width+(py-2)] + (-8.0)*NR[(px-1)*width+(py-1)] + (-12.0)*NR[(px-1)*width+(py)] + (-8.0)*NR[(px-1)*width+(py+1)] + (-2.0)*NR[(px-1)*width+(py+2)]
			+    2.0*NR[(px+1)*width+(py-2)] +    8.0*NR[(px+1)*width+(py-1)] +    12.0*NR[(px+1)*width+(py)] +    8.0*NR[(px+1)*width+(py+1)] +    2.0*NR[(px+1)*width+(py+2)]
			+    1.0*NR[(px+2)*width+(py-2)] +    4.0*NR[(px+2)*width+(py-1)] +     6.0*NR[(px+2)*width+(py)] +    4.0*NR[(px+2)*width+(py+1)] +    1.0*NR[(px+2)*width+(py+2)]);
		
	} else {
		Gx = 
		 (1.0*NR_shared[(i-2)][(j-2)] +  2.0*NR_shared[(i-2)][(j-1)] +  (-2.0)*NR_shared[(i-2)][(j+1)] + (-1.0)*NR_shared[(i-2)][(j+2)]
			+ 4.0*NR_shared[(i-1)][(j-2)] +  8.0*NR_shared[(i-1)][(j-1)] +  (-8.0)*NR_shared[(i-1)][(j+1)] + (-4.0)*NR_shared[(i-1)][(j+2)]
			+ 6.0*NR_shared[(i  )][(j-2)] + 12.0*NR_shared[(i  )][(j-1)] + (-12.0)*NR_shared[(i  )][(j+1)] + (-6.0)*NR_shared[(i  )][(j+2)]
			+ 4.0*NR_shared[(i+1)][(j-2)] +  8.0*NR_shared[(i+1)][(j-1)] +  (-8.0)*NR_shared[(i+1)][(j+1)] + (-4.0)*NR_shared[(i+1)][(j+2)]
			+ 1.0*NR_shared[(i+2)][(j-2)] +  2.0*NR_shared[(i+2)][(j-1)] +  (-2.0)*NR_shared[(i+2)][(j+1)] + (-1.0)*NR_shared[(i+2)][(j+2)]);

		Gy = 
			 ((-1.0)*NR_shared[(i-2)][(j-2)] + (-4.0)*NR_shared[(i-2)][(j-1)] +  (-6.0)*NR_shared[(i-2)][(j)] + (-4.0)*NR_shared[(i-2)][(j+1)] + (-1.0)*NR_shared[(i-2)][(j+2)]
			+ (-2.0)*NR_shared[(i-1)][(j-2)] + (-8.0)*NR_shared[(i-1)][(j-1)] + (-12.0)*NR_shared[(i-1)][(j)] + (-8.0)*NR_shared[(i-1)][(j+1)] + (-2.0)*NR_shared[(i-1)][(j+2)]
			+    2.0*NR_shared[(i+1)][(j-2)] +    8.0*NR_shared[(i+1)][(j-1)] +    12.0*NR_shared[(i+1)][(j)] +    8.0*NR_shared[(i+1)][(j+1)] +    2.0*NR_shared[(i+1)][(j+2)]
			+    1.0*NR_shared[(i+2)][(j-2)] +    4.0*NR_shared[(i+2)][(j-1)] +     6.0*NR_shared[(i+2)][(j)] +    4.0*NR_shared[(i+2)][(j+1)] +    1.0*NR_shared[(i+2)][(j+2)]);
	}

	

	G[imagePixel]   = sqrtf((Gx*Gx)+(Gy*Gy));	//G = √Gx²+Gy²
	phi[imagePixel] = atan2f(fabs(Gy),fabs(Gx));

	if(fabs(phi[imagePixel])<=PI/8 )
		phi[imagePixel] = 0;
	else if (fabs(phi[imagePixel])<= 3*(PI/8))
		phi[imagePixel] = 45;
	else if (fabs(phi[imagePixel]) <= 5*(PI/8))
		phi[imagePixel] = 90;
	else if (fabs(phi[imagePixel]) <= 7*(PI/8))
		phi[imagePixel] = 135;
	else phi[imagePixel] = 0;
}

__global__ void edgesGPU(int *pedge, float *phi, float *G, float *image_out, int level, int height, int width) {

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int i = threadIdx.x, j = threadIdx.y;
	int imagePixel = px*width+py;
	float lowthres, hithres;
	int ii, jj;

	__shared__ float G_shared[CUDABLOCKS][CUDABLOCKS];

	if (px < 0 || px > height || py < 0 || py > width) {
		return;
	}
	G_shared[i][j] = G[imagePixel];
	__syncthreads();

	image_out[imagePixel] = 0;
	pedge[imagePixel] = 0;

	if (px < 3 || px > height-3 || py < 3 || py > width-3) {
		return;
        }

	if (i < 1 || i >= blockDim.x-1 || j < 1 || j >= blockDim.y-1) {

		if(phi[imagePixel] == 0){
			if(G[imagePixel]>G[imagePixel+1] && G[imagePixel]>G[imagePixel-1]) //edge is in N-S
				pedge[imagePixel] = 1;

		} else if(phi[imagePixel] == 45) {
			if(G[imagePixel]>G[(px+1)*width+py+1] && G[imagePixel]>G[(px-1)*width+py-1]) // edge is in NW-SE
				pedge[imagePixel] = 1;

		} else if(phi[imagePixel] == 90) {
			if(G[imagePixel]>G[(px+1)*width+py] && G[imagePixel]>G[(px-1)*width+py]) //edge is in E-W
				pedge[imagePixel] = 1;

		} else if(phi[imagePixel] == 135) {
			if(G[imagePixel]>G[(px+1)*width+py-1] && G[imagePixel]>G[(px-1)*width+py+1]) // edge is in NE-SW
				pedge[imagePixel] = 1;
		}
	} else {
		if(phi[imagePixel] == 0){
			if(G_shared[i][j]>G_shared[i][j+1] && G_shared[i][j]>G_shared[i][j-1]) //edge is in N-S
				pedge[imagePixel] = 1;

		} else if(phi[imagePixel] == 45) {
			if(G_shared[i][j]>G_shared[i+1][j+1] && G_shared[i][j]>G_shared[i-1][j-1]) // edge is in NW-SE
				pedge[imagePixel] = 1;

		} else if(phi[imagePixel] == 90) {
			if(G_shared[i][j]>G_shared[i+1][j] && G_shared[i][j]>G_shared[i-1][j]) //edge is in E-W
				pedge[imagePixel] = 1;

		} else if(phi[imagePixel] == 135) {
			if(G_shared[i][j]>G_shared[i+1][j-1] && G_shared[i][j]>G_shared[i-1][j+1]) // edge is in NE-SW
				pedge[imagePixel] = 1;
		}
	}

	lowthres = level/2;
	hithres  = 2*(level);
	

	if(G_shared[i][j]>hithres && pedge[imagePixel])
		image_out[imagePixel] = 255;
	else if(pedge[imagePixel] && G_shared[i][j]>=lowthres && G_shared[i][j]<hithres)
		// check neighbours 3x3
		for (ii=-1;ii<=1; ii++)
			for (jj=-1;jj<=1; jj++)
				if (i+ii < 0 || j+jj < 0 || i+ii >= blockDim.x || j+jj >= blockDim.y) {
					if (G[(px+ii)*width+(py+jj)]>hithres)
						image_out[imagePixel] = 255;
				} else {
					if (G_shared[(i+ii)][j+jj]>hithres)
						image_out[imagePixel] = 255;
				}
}


void cannyGPU(float *im, float *image_out, float level, int height, int width) {

	float *im_GPU, *image_out_blurred_GPU;
        float *image_out_intensitygradient_G_GPU, *image_out_intensitygradient_phi_GPU, *image_out_GPU;
	int *pedge_GPU;


	/* Mallocs GPU */
	cudaMalloc(&im_GPU, sizeof(float)*height*width);
	cudaMalloc(&image_out_blurred_GPU, sizeof(float)*height*width);

	cudaMalloc(&image_out_intensitygradient_G_GPU, sizeof(float)*height*width);
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
	intensitygradientGPU<<<dimGrid,dimBlock>>>(image_out_blurred_GPU, image_out_intensitygradient_G_GPU, image_out_intensitygradient_phi_GPU, height, width);
	cudaThreadSynchronize();
	edgesGPU<<<dimGrid,dimBlock>>>(pedge_GPU, image_out_intensitygradient_phi_GPU, image_out_intensitygradient_G_GPU, image_out_GPU, level, height, width);
	cudaThreadSynchronize();


	/* GPU->CPU */
	cudaMemcpy(image_out, image_out_GPU, sizeof(float)*height*width, cudaMemcpyDeviceToHost);


	cudaFree(image_out_blurred_GPU);
	cudaFree(image_out_intensitygradient_G_GPU);
	cudaFree(image_out_intensitygradient_phi_GPU);
	cudaFree(image_out_GPU);
	cudaFree(pedge_GPU);

}






