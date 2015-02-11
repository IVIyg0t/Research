
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"
#include "cpu_anim.h"
#include "cuda.h"
#include <stdio.h>

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernal(unsigned char *ptr, int ticks){
	//map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	//Clalculate the value at that position
	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

	ptr[offset * 4 + 0] = grey;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

struct DataBlock{
	unsigned char *dev_bitmap;
	CPUAnimBitmap
};

void generate_frame(DataBlock *d, int ticks){
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	kernal << <blocks, threads >> >(d->dev_bitmap, ticks);

	HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(),
							d->dev_bitmap,
							d->bitmap->image_size(),
							cudaMemcpyDeviceToHost));
}

//Clean up memory allocated on the GPU
void cleanup(DataBlock *d) {
	cudaFree(d->dev_bitmap);
}

int main(void){
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);

	HANDLE_ERROR(cudaMalloc((void **)&data.dev_bitmap, bitmap.image_size()));

	bitmap.anim_and_exit((void(*)(void*, int))generate_frame, (void(*)(void*))cleanup);

}
