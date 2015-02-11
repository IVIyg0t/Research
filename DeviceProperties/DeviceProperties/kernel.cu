
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "..\..\cuda_by_example\common\book.h"

int main(void) {
	cudaDeviceProp  dev;

	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for (int i = 0; i< count; i++) {
		HANDLE_ERROR(cudaGetDeviceProperties(&dev, i));
		printf("   --- General Info about Device %d ---\n", i);
		printf("Name:  %s\n",dev.name);
		printf("Compute capability:  %d.%d\n",dev.major,dev.minor);
		printf("Clock rate:  %d\n", dev.clockRate);
		printf("Device copy overlap:  ");
		if (dev.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execution timeout :  ");
		if (dev.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		printf("   --- Memory Info for device %d ---\n", i);
		printf("Total global mem:  %ld\n", dev.totalGlobalMem);
		printf("Total constant Mem:  %ld\n", dev.totalConstMem);
		printf("Max mem pitch:  %ld\n", dev.memPitch);
		printf("Texture Alignment:  %ld\n", dev.textureAlignment);

		printf("   --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count:  %d\n",dev.multiProcessorCount);
		printf("Shared mem per mp:  %ld\n", dev.sharedMemPerBlock);
		printf("Registers per mp:  %d\n", dev.regsPerBlock);
		printf("Threads in warp:  %d\n", dev.warpSize);
		printf("Max threads per block:  %d\n",dev.maxThreadsPerBlock);
		printf("Max thread dimensions:  (%d, %d, %d)\n",dev.maxThreadsDim[0], dev.maxThreadsDim[1],dev.maxThreadsDim[2]);
		printf("Max grid dimensions:  (%d, %d, %d)\n",dev.maxGridSize[0], dev.maxGridSize[1],dev.maxGridSize[2]);
		printf("\n");
	}
	char done = 'n';
	printf("done?");
	scanf("%c", done);
}