
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"
#include "cpu_bitmap.h"
#include <stdio.h>

//Code for Juliet Set
#define DIM 1000

struct cuComplex {
	float r;
	float i;
	 __host__ __device__ cuComplex(float a, float b) : r(a), i(b) {}

	__device__ float magnitude2(void){
		return r*r + i*i;
	}

	__device__ cuComplex operator*(const cuComplex& a){
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a){
		return cuComplex(r + a.r, i + a.i);
	}
};


__device__ int julia(int x, int y){
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.81, -0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++){
		a = a*a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}

__global__ void kernal(unsigned char *ptr) {
	//map from threadIdx/Blockidx to pixel position
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	// Now calculate the value at that position
	int juliaValue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main(void) {
	CPUBitmap bitmap(DIM, DIM);

	unsigned char *dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));
	
	dim3 grid(DIM, DIM);

	kernal<<<grid, 1>>>(dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
}



//Source code for vector addition
/*
#define N 10

__global__ void add(const int *a, const int *b, int *c){
	int tid = blockIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void) {
	int host_a[N], host_b[N], host_c[N];
	int *dev_a, *dev_b, *dev_c;
	
	//Allocate memory on GPU
	HANDLE_ERROR(cudaMalloc((void **)&dev_a, N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_b, N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_c, N*sizeof(int)));

	//Fill host arrays with some values
	for (int i = 0; i < N; i++) {
		host_a[i] = -i;
		host_b[i] = i * i;
	}

	for (int i = 0; i < N; i++){
		printf("a[%d] = %d    b[%d] = %d\n", i, host_a[i], i, host_b[i]);
	}

	//Copy Host arrays 'a' and 'b' to the GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, host_a, N*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, host_b, N*sizeof(int), cudaMemcpyHostToDevice));

	//Run the kernal
	add<<<N, 1>>>(dev_a, dev_b, dev_c);

	//Copy contents of dev_c to host_c
	HANDLE_ERROR(cudaMemcpy(host_c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));

	//Display results
	for (int i = 0; i < N; i++){
		printf("%d + %d = %d\n", host_a[i], host_b[i], host_c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	while (1);

	return 0;
}
*/