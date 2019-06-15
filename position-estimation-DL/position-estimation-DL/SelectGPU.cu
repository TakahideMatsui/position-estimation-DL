#include"SelectGPU.h"

cudaError_t SelectBestGPU(int* device) {
	cudaError_t err;
	int device_count;

	err = cudaGetDeviceCount(&device_count);

	if (err != cudaSuccess) {
		fprintf_s(stderr, "Cannot get device count.");
		return err;
	}

	int *device_ids;

	device_ids = (int*)malloc(sizeof(int) * device_count);

	cudaDeviceProp device_prop;
	int maxClock = 0;
	for (int id = 0; id < device_count; id++) {
		cudaGetDeviceProperties(&device_prop, id);
		if (maxClock < device_prop.clockRate) {
			maxClock = device_prop.clockRate;
			*device = id;
		}
	}
	return cudaSuccess;
}

cudaError_t SelectGPU(int* device) {
	cudaError_t err;
	int device_count;

	err = cudaGetDeviceCount(&device_count);

	if (err != cudaSuccess) {
		fprintf_s(stderr, "Cannot get device count.");
		return err;
	}

	int *device_ids;

	device_ids = (int*)malloc(sizeof(int) * device_count);

	cudaDeviceProp device_prop;

	printf("please chose GPU for Below.\n\n");
	for (int id = 0; id < device_count; id++) {
		cudaGetDeviceProperties(&device_prop, id);
		device_ids[id] = id;
		printf("[%d]:%s\n", id, device_prop.name);
	}

	int id;
	printf("please input id >>");
	scanf("%d", &id);

	*device = device_ids[id];
	return cudaSuccess;
}

cudaError_t SelectGPU(int *device, const char* strDevice) {
	cudaError_t err;
	int device_count;

	err = cudaGetDeviceCount(&device_count);

	if (err != cudaSuccess) {
		fprintf_s(stderr, "Cannot get device count.");
		return err;
	}

	int *device_ids;

	device_ids = (int*)malloc(sizeof(int) * device_count);
	cudaDeviceProp device_prop;

	int hits = 0;
	for (int id = 0; id < device_count; id++) {
		cudaGetDeviceProperties(&device_prop, id);
		if (strcmp(device_prop.name, strDevice) == 0) {
			*device = id;
			return cudaSuccess;
		}
	}
	return cudaErrorInvalidDevice;
}

cudaError_t SelectGPU(int* device, const std::string strDevice) {
	cudaError_t err;
	int device_count;

	err = cudaGetDeviceCount(&device_count);

	if (err != cudaSuccess) {
		fprintf_s(stderr, "Cannot get device count.");
		return err;
	}

	int *device_ids;

	device_ids = (int*)malloc(sizeof(int) * device_count);
	cudaDeviceProp device_prop;

	int hits = 0;
	for (int id = 0; id < device_count; id++) {
		cudaGetDeviceProperties(&device_prop, id);
		if (strcmp(device_prop.name, strDevice.c_str()) == 0) {
			*device = id;
			return cudaSuccess;
		}
	}
	return cudaErrorInvalidDevice;


}