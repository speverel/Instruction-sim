#include <stdlib.h>
#include <stdio.h>
#include "../cudpp/include/cudpp.h"
#include "../cudpp/include/cudpp_config.h"
#define WIDTH 32
#define HEIGHT 32
#define MAX_VAL 120

#define CLEANUP(s, v) \
	do { \
		printf("%s\n", s); \
		if (h_Act)			free(h_Act); \
		if (h_Mat)			free(h_Mat); \
		if (h_Out)			free(h_Out); \
		if (handle)			cudppDestroy(handle); \
		if (data)			cudaFree(data); \
		if (startAddr)		cudaFree(startAddr); \
		if (endAddr)		cudaFree(endAddr); \
		if (*lowestAddr)	cudaFree(*lowestAddr); \
		if (output)			cudaFree(output); \
		if (indices)		cudaFree(indices); \
		if (matrix)			cudaFree(matrix); \
		if (P)				cudaFree(P); \
		cudaDeviceReset(); \
		fflush(stdout); \
	} while (0); \
	return v;
/*
		if (data)			cudaFree(data); \
		if (startAddr)		cudaFree(startAddr); \
		if (endAddr)		cudaFree(endAddr); \
		if (*lowestAddr)	cudaFree(*lowestAddr); \
		if (output)			cudaFree(output); \
		if (indices)		cudaFree(indices); \
		if (matrix)			cudaFree(matrix); \
		if (P)				cudaFree(P); \
*/
__global__ void setIndices(int* indices)
{
	unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
	indices[tid] = tid;
}

__global__ void findLowest(float** result, float* data, int bias)
{
	unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (bias == 0 || tid < blockDim.x * gridDim.x - 1) // prevents last thread from accessing off the end
	{
		if (data[2 * tid + bias] == 0 && data[2 * tid + 1 + bias] > 0) // this is the zero boundary if true
			*result = data + 2 * tid + 1 + bias;	
	}
}

// launch endAddr(a) - startAddr(a) blocks of height threads each
__global__ void denseSparseMVM(float* W, float* a, int* ind, unsigned height, unsigned width, float* P)
{
	unsigned tid = threadIdx.x;
	unsigned i = blockIdx.x;
	float* pw = W + ind[i] * height;
	P[i * height + tid] = pw[tid] * a[i];
}
/*
__global__ void sparseSparseMVM(float* W, float** Wends, int* Windices, float* a, float* aend, int* aindices, unsigned height, float** P)
{
	extern __shared__ int* Pindices;
	unsigned tid = threadIdx.x;
	for (unsigned i = 0; i <= (aend - a); i++)
	{
		int aindex = aindices[i];
		float* colEnd = Wends[aindex];
		float* colBeg = W + aindex * height;
		if (colBeg + tid <= colEnd)
		{
			int* Wcolind = Windices + aindex * height;
			int windex = Wcolind[tid];
			float psum = colBeg[tid] * a[i];
			unsigned pI = Pindices[windex];
			P[windex][pI] = psum;
			Pindices[windex] = pI + 1;
		}
	}
}	
*/
void merge(int low, int mid, int high, float* idata, float* odata, int* iind, int* oind)
{
	int l1, l2, i;
	for (l1 = low, l2 = mid + 1, i = low; l1 <= mid && l2 <= high; i++)
	{   
		if (idata[l1] <= idata[l2])
		{
			oind[i] = iind[l1];
			odata[i] = idata[l1++];
		}
		else
		{
			oind[i] = iind[l2];
			odata[i] = idata[l2++];
		}
	}
	while (l1 <= mid)
	{
		oind[i] = iind[l1];
		odata[i++] = idata[l1++];
	}
	while (l2 <= high)
	{
		oind[i] = iind[l2];
		odata[i++] = idata[l2++];
	}
	for (i = low; i <= high; i++)
	{
		idata[i] = odata[i];
		iind[i] = oind[i];
	}
}

void mergesort(int low, int high, float* idata, float* odata, int* iind, int* oind)
{
	int mid;
	if (low < high)
	{
		mid = (low + high) / 2;
		mergesort(low, mid, idata, odata, iind, oind);
		mergesort(mid + 1, high, idata, odata, iind, oind);
		merge(low, mid, high, idata, odata, iind, oind);
	}
	else
		return;
}

void printGold(float* vector, int length, float* dataPtr)
{
	float* tempData = (float*)malloc(sizeof(float) * length);
	int* indices = (int*)malloc(sizeof(int) * length);
	int* tempIndices = (int*)malloc(sizeof(int) * length);
	for (int i = 0; i < length; i++)
		indices[i] = i;
	float *startAddr, *endAddr;
	endAddr = dataPtr + length;
	mergesort(0, length, vector, tempData, indices, tempIndices);
	int index = 0;
	while (tempData[index] == 0.0)
		index++;
	startAddr = dataPtr + index;
	printf("CPU results:\n");
	printf("Start pointer: %p End pointer: %p\n", startAddr - 1, endAddr);
	for (int i = startAddr - dataPtr; i <= endAddr - dataPtr; i++)
		printf("Value: %f Index: %d\n", tempData[i], tempIndices[i]);
	free(tempData);
	free(indices);
	free(tempIndices);
}

int main()
{
	srand(time(NULL));
	float* h_Act = (float*)malloc(sizeof(float) * WIDTH);
	float* h_Mat = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);
	float* h_Out = (float*)malloc(sizeof(float) * HEIGHT);
	float temp;
	CUDPPHandle handle = 0;
	CUDPPHandle scanplan = 0;
	int nnz = 0;
	for (int i = 0; i < WIDTH; i++) // generate random floats, which are rectified
	{
		temp = (float)rand() / (float)(RAND_MAX / MAX_VAL) - MAX_VAL / 2.5;
		if (temp > 0)
		{
			h_Act[i] = temp;
			nnz++;
		}
		else
		{
			h_Act[i] = 0;
		}
	}

	// populate matrix
	for (int i = 0; i < WIDTH; i++)
	{
		for (int j = 0; j < HEIGHT; j++)
		{
			h_Mat[i * HEIGHT + j] = (float)rand() / (float)(RAND_MAX / (MAX_VAL)) - MAX_VAL / 2;
		}
	}
	// allocate memory on GPU, and copy activation vector over
	float *data, *output, *matrix, *startAddr, *endAddr, *P;
	int* indices;
	float** foo;
	float*** lowestAddr = &foo;
	cudaError_t err1 = cudaMallocManaged(&data, sizeof(float) * WIDTH, cudaMemAttachGlobal);
	cudaError_t err2 = cudaMallocManaged(&indices, sizeof(int) * WIDTH, cudaMemAttachGlobal);
	cudaError_t err3 = cudaMallocManaged(&startAddr, sizeof(float), cudaMemAttachGlobal);
	cudaError_t err5 = cudaMalloc(&matrix, sizeof(float) * HEIGHT * WIDTH);
	cudaError_t err6 = cudaMallocManaged(lowestAddr, sizeof(float**), cudaMemAttachGlobal);
	cudaError_t err7 = cudaMallocManaged(&endAddr, sizeof(float), cudaMemAttachGlobal);
	cudaError_t err8 = cudaMallocManaged(&P, sizeof(float) * HEIGHT * WIDTH);
	if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess) 
	{
		CLEANUP("Failed to allocate memory on device.", 1);
	}
	err1 = cudaMemcpy(data, h_Act, sizeof(float) * WIDTH, cudaMemcpyHostToDevice);
	err3 = cudaMemcpy(matrix, h_Mat, sizeof(float) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
	if (err1 != cudaSuccess || err3 != cudaSuccess)
	{
		CLEANUP("Failed to copy memory to device.", 1);
	}

	printGold(h_Act, WIDTH, data);

	setIndices<<<8, WIDTH / 8>>>(indices);

	cudppCreate(&handle);
	CUDPPConfiguration config;
	config.algorithm = CUDPP_SORT_RADIX;
	config.datatype = CUDPP_FLOAT;
	config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

	CUDPPResult res = cudppPlan(handle, &scanplan, config, WIDTH, 1, 0);

	if (res != CUDPP_SUCCESS)
	{
		CLEANUP("Failed to create plan.", 1);
	}

	res = cudppRadixSort(scanplan, data, indices, WIDTH);
	
	if (res != CUDPP_SUCCESS)
	{
		CLEANUP("Failed to execute radix sort.", 1);
	}	
	float* initLow = **lowestAddr;
	cudaDeviceSynchronize();
	findLowest<<<8, WIDTH / 16>>>(*lowestAddr, data, 0);
	cudaDeviceSynchronize();
	if (**lowestAddr == initLow)
	{
		findLowest<<<8, WIDTH / 16>>>(*lowestAddr, data, 1);
		cudaDeviceSynchronize();
	}
	if (**lowestAddr == initLow)
	{
		CLEANUP("Could not find zero boundary.", 1);
	}
	startAddr = **lowestAddr;
	endAddr = data + WIDTH;
		
	// perform dense sparse matrix vector multiplication, using the above sparse vector
	denseSparseMVM<<<endAddr - startAddr, HEIGHT>>>(matrix, startAddr, indices + (startAddr - data), HEIGHT, WIDTH, P);
	// reduce psums into result
	CUDPPConfiguration configReduce;
	configReduce.algorithm = CUDPP_REDUCE;
	configReduce.op = CUDPP_ADD;
	configReduce.datatype = CUDPP_FLOAT;

	cudaError_t err4 = cudaMalloc(&output, sizeof(float) * HEIGHT);
	if (err4 != cudaSuccess)
	{
		CLEANUP("Failed to allocate memory on the device.", 1);
	}	
	res = cudppPlan(handle, &scanplan, configReduce, WIDTH, 1, 0);
	if (res != CUDPP_SUCCESS)
	{
		CLEANUP("Failed to create plan.", 1);
	}

	for (int i = 0; i < HEIGHT; i++)
	{
		res = cudppReduce(scanplan, output + i, P + i * WIDTH, (endAddr - startAddr)); 
		if (res != CUDPP_SUCCESS)
		{
			CLEANUP("Failed to reduce psums.", 1);
		}
	}
	
	cudaDeviceSynchronize();
	err1 = cudaMemcpy(h_Out, output, sizeof(float) * HEIGHT, cudaMemcpyDeviceToHost);
	if (err1 != cudaSuccess)
	{
		CLEANUP("Failed to copy data back to host.", 1);
	}
	printf("-------------------------------------\nGPU Results:\n");		
	printf("Start pointer: %p End pointer: %p\n", startAddr, endAddr);
	for (int i = startAddr - data; i < endAddr - data; i++)
	{
		printf("Input: %f Index: %d\n", data[i], indices[i]);
	}
	for (int j = 0; j < HEIGHT; j++)
	{
		printf("Output: %f\n", h_Out[j]);
	}
	cudppDestroyPlan(scanplan);
	CLEANUP("Program completed successfully.", 0);
}
