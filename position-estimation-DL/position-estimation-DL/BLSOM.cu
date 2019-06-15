
#include "BLSOM.h"
#include "SelectGPU.h"
#include<algorithm>
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

#ifndef DIST
#define DIST(bx,by,x,y) ((bx-x)*(bx-x)+(by-y)*(by-y))
#endif // !DIST


#ifndef MAX
#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#define CHECK(call)														\
{																		\
	const cudaError_t error = call;										\
	if(error!=cudaSuccess){												\
		printf("Error %s:%d \t",__FILE__,__LINE__);						\
		printf("code:%d, reason:%s\n",error,cudaGetErrorString(error));	\
		exit(1);														\
	}																	\
}


bool checkAllocatedMemory(void* pointer) {
	if (pointer != NULL) {
		return true;
	}
	else {
		return false;
	}
}

BLSOM::BLSOM(int vec_dim, int map_width) :iAlfa(0.5), iBeta(40), t_alfa(30), t_beta(20),
										  vec_dim(vec_dim), map_width(map_width), flg_gpu(true), flg_iniBatch(false) {
	int device;
	
	this->map_height = 0;
	CHECK(SelectBestGPU(&device));

	if (flg_gpu) {
		CHECK(cudaSetDevice(device));
	}
}

BLSOM::BLSOM(int vec_dim, int map_width, int map_height) :iAlfa(0.5), iBeta(40), t_alfa(30), t_beta(20),
														  vec_dim(vec_dim), map_width(map_width), map_height(map_height), flg_gpu(true), flg_iniBatch(false) {
	int device;
	CHECK(SelectBestGPU(&device));

	if (flg_gpu) {
		CHECK(cudaSetDevice(device));
	}
}

BLSOM::BLSOM(int vec_dim, int map_width, int map_height,int device):iAlfa(0.5), iBeta(40), t_alfa(30), t_beta(20), 
																    vec_dim(vec_dim), map_width(map_width), map_height(map_height),flg_gpu(true), flg_iniBatch(false) {
	if (flg_gpu) {
		CHECK(cudaSetDevice(device));
	}
}

BLSOM::BLSOM(int vec_dim, int map_width, int map_height, int device, int gpuFlag) : iAlfa(0.5), iBeta(40), t_alfa(30), t_beta(20), 
																					vec_dim(vec_dim),map_width(map_width),map_height(map_height),flg_gpu(gpuFlag), flg_iniBatch(false) {
	
	if (gpuFlag) {
		CHECK(cudaSetDevice(device));
	}
}

BLSOM::~BLSOM() {
	
}

void BLSOM::Init(const float sdev1, const float sdev2, const float* rot1, const float* rot2, const float *aveVec) {

	if (map_height == 0) {
		this->map_height = (sdev2 / sdev1)*this->map_width;
	}

	if (flg_gpu) {
		
		this->d_mapWeight = thrust::device_vector<float>(map_width*map_height*vec_dim);
		this->d_weightS = thrust::device_vector<float>(map_width*map_height* (vec_dim));
		this->d_cntWeightS = thrust::device_vector<float>(map_width*map_height);
		this->d_node = thrust::device_vector<float>(map_width*map_height);
		this->d_rot1 = thrust::device_vector<float>(vec_dim);
		this->d_rot2 = thrust::device_vector<float>(vec_dim);
		this->d_aveVec = thrust::device_vector<float>(vec_dim);
		this->d_sdev = thrust::device_vector<float>(2);
		this->d_bmuPos = thrust::device_vector<int>(2);

		cudaMemcpy(thrust::raw_pointer_cast(this->d_rot1.data()), rot1, this->vec_dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(thrust::raw_pointer_cast(this->d_rot2.data()), rot2, this->vec_dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(thrust::raw_pointer_cast(this->d_aveVec.data()), aveVec, this->vec_dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(thrust::raw_pointer_cast(this->d_sdev.data()), &sdev1, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(thrust::raw_pointer_cast(this->d_sdev.data()+1), &sdev2, sizeof(float), cudaMemcpyHostToDevice);
	}

	this->h_mapWeight = thrust::host_vector<float>(this->map_width*this->map_height*this->vec_dim);
	this->h_weightS = thrust::host_vector<float>(this->map_width*this->map_height* (this->vec_dim));
	this->h_cntWeightS = thrust::device_vector<float>(map_width*map_height);
	this->h_node = thrust::host_vector<float>(this->map_width*this->map_height);
	this->h_rot1 = thrust::host_vector<float>(this->vec_dim);
	this->h_rot2 = thrust::host_vector<float>(this->vec_dim);
	this->h_aveVec = thrust::host_vector<float>(this->vec_dim);
	this->h_sdev = thrust::host_vector<float>(2);
	this->h_bmuPos = thrust::host_vector<int>(2);

	memcpy(thrust::raw_pointer_cast(this->h_rot1.data()), rot1, this->vec_dim * sizeof(float));
	memcpy(thrust::raw_pointer_cast(this->h_rot2.data()), rot2, this->vec_dim * sizeof(float));
	memcpy(thrust::raw_pointer_cast(this->h_aveVec.data()), aveVec, this->vec_dim * sizeof(float));
	memcpy(thrust::raw_pointer_cast(this->h_sdev.data()), &sdev1, sizeof(float));
	memcpy(thrust::raw_pointer_cast(this->h_sdev.data()+1), &sdev2, sizeof(float));

	this->flg_iniBatch = true;
}

void BLSOM::SetTrainingData(const std::vector<std::vector<float>> train) {
	float* tempTrain;
	float* temp_begin;

	this->train_num = train.size();
	this->vec_dim = train[0].size();

	tempTrain = new float[train_num*this->vec_dim];
	temp_begin = tempTrain;

	thrust::host_vector<float> h_train = thrust::host_vector<float>(train_num*this->vec_dim);
	thrust::device_vector<float> d_train = thrust::device_vector<float>(train_num*this->vec_dim);

	for_each(train.begin(), train.end(), [&](std::vector<float> data) {memcpy(tempTrain, data.data(), data.size() * sizeof(float)); tempTrain += data.size();});

	memcpy(thrust::raw_pointer_cast(h_train.data()), temp_begin,train_num*this->vec_dim * sizeof(float));
	cudaMemcpy(thrust::raw_pointer_cast(d_train.data()), thrust::raw_pointer_cast(h_train.data()), train_num*this->vec_dim * sizeof(float), cudaMemcpyHostToDevice);
	this->d_trains.push_back(d_train);

	free(temp_begin);

}

void BLSOM::SetTrainingData(const std::vector<std::vector<std::vector<float>>> train) {
	
	for (auto trains: train) {
		this->SetTrainingData(trains);
	}
}


void BLSOM::check_mapWeight() {
	cudaMemcpy(thrust::raw_pointer_cast(this->h_mapWeight.data()), thrust::raw_pointer_cast(this->d_mapWeight.data()), sizeof(float)*this->map_width*this->map_height*this->vec_dim, cudaMemcpyDeviceToHost);

	for (int idy = 0; idy < map_height; idy++) {
		for (int idx = 0; idx < map_width; idx++) {
			//printf("%d %d \n",idy, idx);
			printf("%d", map_width*idy + idx);
			//printf("%d", map_width*vec_dim*idy + vec_dim*idx);
			/*
			for (int idz = 0; idz < vec_dim; idz++) {
				printf("%d :", map_width*vec_dim*idy + vec_dim*idx + idz);
				printf("%f ", this->h_mapWeight[map_width*vec_dim*idy + vec_dim*idx + idz]);
				printf("\n");
			}*/
			printf("\n");
		}
	}
}

__global__ void InitMapWeightFromGPU(float* mapWeight,float* ave_vec, float* sdev, float* rot1, float* rot2, const int map_width, const int map_height, const int vec_dim) {
	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int idx = map_width*vec_dim*iy + vec_dim*ix + threadIdx.z;

	float sigmaB1 = 5 * sdev[0] * rot1[threadIdx.z];
	float sigmaB2 = 5 * sdev[1] * rot2[threadIdx.z];

	mapWeight[idx] = ave_vec[threadIdx.z]+ sigmaB1*((ix - (map_width / 2.0)) / map_width) + sigmaB2*((iy - (map_height / 2.0)) / map_height);
}

__global__ void setup_kernel(curandState *state, const int map_width, const int vec_dim)
{
	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int idx = map_width*vec_dim*iy + vec_dim*ix + threadIdx.z;
	curand_init(1234, idx, 0, &state[idx]);
}

__global__ void InitMapWeightRandFromGPU(curandState* state, float* mapWeight, const int map_width, const int vec_dim, unsigned int l, unsigned int h) {
	unsigned int x;
	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int idx = map_width*vec_dim*iy + vec_dim*ix + threadIdx.z;

	curandState  localState = state[idx];
	x = curand(&localState);

	state[idx] = localState;
	mapWeight[idx] = l + (x % (h+1));
}

void BLSOM::InitMapWeightRand() {

	dim3 block(1, 1, this->vec_dim);
	dim3 grid(this->map_width, this->map_height);
	thrust::device_vector<curandState> devStates(this->map_width * this->map_height * this->vec_dim);

	setup_kernel <<< grid, block >>> (thrust::raw_pointer_cast(devStates.data()),this->map_width, this->vec_dim);
	InitMapWeightRandFromGPU <<< grid, block >>>(thrust::raw_pointer_cast(devStates.data()),
		thrust::raw_pointer_cast(this->d_mapWeight.data()),
		this->map_width,
		this->vec_dim,
		0,
		255);
}

void BLSOM::InitMapWeightBatch() {
	dim3 block(1, 1, this->vec_dim);
	dim3 grid(this->map_width, this->map_height);

	if (this->flg_iniBatch) {
		InitMapWeightFromGPU << < grid, block >> > (thrust::raw_pointer_cast(this->d_mapWeight.data()),
													thrust::raw_pointer_cast(this->d_aveVec.data()),
													thrust::raw_pointer_cast(this->d_sdev.data()),
													thrust::raw_pointer_cast(this->d_rot1.data()),
													thrust::raw_pointer_cast(this->d_rot2.data()),
													this->map_width,
													this->map_height,
													this->vec_dim);
	}
	else {
		std::cerr << "Please call BLSOM::Init, before call InitMapWeightBatch." << std::endl;
	}
}

void BLSOM::InitMapWeight(int mode) {
	

	switch (mode){
	case INIT_BATCH:
		InitMapWeightBatch();
		break;

	case INIT_RANDOM:
		InitMapWeightRand();
		break;

	default:
		break;
	}
	
}

__global__ void InitNodeFromGPU(float* node,const int map_width) {
	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int idx = map_width*iy + ix;
	node[idx] = 0;
}

__global__ void BMUFromGPU(float* input_xk, float* node, float* mapWeight, const int map_width, const int vec_dim) {
	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int node_idx = map_width*iy + ix;
	int map_idx = map_width*vec_dim*iy + vec_dim*ix;// + threadIdx.z;
	
	for(int dim=0;dim<vec_dim;dim++)
		node[node_idx] += (mapWeight[map_idx+dim]-input_xk[dim])*(mapWeight[map_idx + dim] - input_xk[dim]);
	
}

int BLSOM::getBMUIndex() {
	thrust::device_vector<float>::iterator bgn_itr = d_node.begin();
	thrust::device_vector<float>::iterator bmu_itr = thrust::min_element(thrust::device, d_node.begin(), d_node.end());
	return thrust::distance(bgn_itr, bmu_itr);
}

void BLSOM::setBMUPosition() {
	int bmu_index = getBMUIndex();
	this->h_bmuPos[0] = bmu_index % (this->map_width);	//xÀ•WŒvŽZ
	this->h_bmuPos[1] = bmu_index / (this->map_width);	//yÀ•WŒvŽZ
	this->d_bmuPos = this->h_bmuPos;
}

__global__ void CalcWeightSFromGPU(float* input_xk, int* bmuPos, float* weightS,float* cntWeightS,
								   const int map_width, const int vec_dim,
								   const double iBeta, const double tBeta, const int lnum) {

	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int weiS_idx = map_width*vec_dim*iy + vec_dim*ix;// +threadIdx.z;
	int cntS_idx = map_width*iy + ix;				//cntWeightS[ix][iy]


	float dist = DIST(bmuPos[0], bmuPos[1], ix, iy);
	float Beta = MAX(0, (iBeta*(1 - (lnum / tBeta))));

	if ((Beta*Beta - dist) >= 0) {
		//printf("calsWeightS\n");
		for (int dim = 0; dim < vec_dim; dim++) {
			weightS[weiS_idx + dim] += input_xk[dim];
		}
		cntWeightS[cntS_idx]++;
	}

}

__global__ void UpdateMapWeightFromGPU(float* mapWeight, float* weightS, float* cntWeightS,
									   const int map_width, const int vec_dim,
									   const double iAlfa, const double tAlfa, const int lnum) {
	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int map_idx = map_width*vec_dim*iy + vec_dim*ix;// +threadIdx.z;
	int cntS_idx = map_width*iy + ix;							//weightS[ix][iy][vec_dim]

	float alfaFunc = MAX(0.01, (iAlfa*(1.0 - (lnum / tAlfa))));

	if (cntWeightS[cntS_idx] > 0) {
		for (int dim = 0; dim < vec_dim; dim++) {
			weightS[map_idx + dim] /= cntWeightS[cntS_idx];
			weightS[map_idx + dim] -= mapWeight[map_idx + dim];
			weightS[map_idx + dim] *= alfaFunc;
			mapWeight[map_idx + dim] += weightS[map_idx + dim];
		}
	}
}

void BLSOM::BMU(float* input_xk) {
	dim3 block(1, 1, this->vec_dim);
	dim3 grid(this->map_width, this->map_height);

	InitNodeFromGPU <<< grid, 1 >>> (thrust::raw_pointer_cast(this->d_node.data()),this->map_width);
	BMUFromGPU <<< grid,1 >>>(input_xk, thrust::raw_pointer_cast(this->d_node.data()), thrust::raw_pointer_cast(this->d_mapWeight.data()), this->map_width, this->vec_dim);
	setBMUPosition();
	
}

void BLSOM::CalcWeightS(float* input_xk, int Lnum) {
	dim3 block(1, 1, this->vec_dim);
	dim3 grid(this->map_width, this->map_height);

	CalcWeightSFromGPU <<<grid, 1 >>> (input_xk,
										   thrust::raw_pointer_cast(this->d_bmuPos.data()),
										   thrust::raw_pointer_cast(this->d_weightS.data()),
										   thrust::raw_pointer_cast(this->d_cntWeightS.data()),			   
										   this->map_width,
										   this->vec_dim,
										   this->iBeta,
										   this->t_beta,
										   Lnum);
											
}

void BLSOM::UpdateMapWeight(int Lnum) {
	dim3 block(1, 1, this->vec_dim);
	dim3 grid(this->map_width, this->map_height);

	UpdateMapWeightFromGPU <<<grid,1>>> (thrust::raw_pointer_cast(this->d_mapWeight.data()),
											 thrust::raw_pointer_cast(this->d_weightS.data()),
											 thrust::raw_pointer_cast(this->d_cntWeightS.data()),
											 
											 this->map_width,
											 this->vec_dim,
											 this->iAlfa,
											 this->t_alfa,
											 Lnum);
}

__global__ void InitCntWeightSFromGPU(float* cntWeightS) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	cntWeightS[idx] = 0;

}

__global__ void InitWeighSFromGPU(float* weightS) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	weightS[idx] = 0;
}

void BLSOM::Learning(int Lnum) {
	std::cout << "Learning Start" << std::endl;
	
	int nWeightS = this->map_width*this->map_height*this->vec_dim;
	dim3 weightS_block(this->vec_dim);
	dim3 weightS_grid((nWeightS+weightS_block.x-1)/weightS_block.x);

	int nCntWeightS = this->map_width*this->map_height;
	dim3 cntWeightS_block(this->map_height);
	dim3 cntWeightS_grid((nCntWeightS+cntWeightS_block.x-1)/cntWeightS_block.x);


	for (int l = 0; l < Lnum; l++) {
		std::cout << "Learning : " << l << "/" << Lnum << "\r";

		for (int i = 0; i < this->d_trains.size(); i++) {
			
			InitCntWeightSFromGPU << <cntWeightS_grid, cntWeightS_block >> > (thrust::raw_pointer_cast(this->d_cntWeightS.data()));
			InitWeighSFromGPU << <weightS_grid, weightS_block >> > (thrust::raw_pointer_cast(this->d_weightS.data()));

			for (int j = 0; j < this->train_num; j++) {
				this->BMU(thrust::raw_pointer_cast(&(this->d_trains[i][j*(this->vec_dim)]))); //“Y‚¦Žš‚ðC³
				this->CalcWeightS(thrust::raw_pointer_cast(&(this->d_trains[i][j*(this->vec_dim)])), l);
			}
			this->UpdateMapWeight(l);
		}
	}

	std::cout << "Learning Finish" << std::endl;
}

float* BLSOM::GetSOMMap() {
	this->h_mapWeight = this->d_mapWeight;
	return thrust::raw_pointer_cast(this->h_mapWeight.data());
}

void BLSOM::d_showWeightS() {
	for (int h = 0; h < this->map_height; h++) {
		for (int w = 0; w < this->map_width; w++) {
			std::cout << "(" << w << "," << h << "): ";
			for (int d = 0; d < this->vec_dim; d++) {
				 std::cout << this->d_weightS[h*map_width*vec_dim + w*vec_dim + d] << " ";
			}
			std::cout << "\n";
		}
	}
}

void BLSOM::d_showMapWeight() {
	for (int h = 0; h < this->map_height; h++) {
		for (int w = 0; w < this->map_width; w++) {
			std::cout << "(" << w << "," << h << "): ";
			for (int d = 0; d < this->vec_dim; d++) {
				std::cout << this->d_mapWeight[h*map_width*vec_dim + w*vec_dim + d] << " ";
			}
			std::cout << "\n";
		}
	}
}