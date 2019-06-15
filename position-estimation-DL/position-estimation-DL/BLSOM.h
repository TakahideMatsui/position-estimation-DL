#pragma once

#include<iostream>
#include<string>
#include<cmath>
#include<vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include<thrust\device_vector.h>
#include<thrust\host_vector.h>
#include<thrust\extrema.h>
#include <thrust/execution_policy.h>
#include<memory>

#define INIT_BATCH 0
#define INIT_RANDOM 1

const int InitBatch = 0;
const int InitRandom = 1;

class BLSOM {
private:

	/*--- マップの基本情報 ---*/
	int epoc_num;	//エポック数
	int train_num;	//学習データの数
	int vec_dim;	//対象データの次元
	int map_width;	//マップの横幅I
	int map_height;	//マップの高さJ

	/*--- 学習係数と近傍関数のparameter ---*/
	double t_alfa;	//学習係数
	double t_beta;	//近傍関数
	double iAlfa;	//初期値α
	double iBeta;	//初期値β

	/*--- GPU用計算変数ここから ---*/
	///BLSOMに利用
	bool flg_gpu;								//GPUを使用するか否か
	bool flg_iniBatch;
	int   d_Acs;							//使用するGPU
	thrust::device_vector<float> d_mapWeight;						//J×I行列のマップの領域確保に利用
	thrust::device_vector<float> d_weightS;							//代表ベクトルWij
	thrust::device_vector<float> d_cntWeightS;						//代表ベクトルの分類数
	thrust::device_vector<float> d_node;							//node
	thrust::device_vector<int> d_bmuPos;							//d_bmuPos[0]=x,d_bmuPos[1]=y
	//thrust::device_vector<float> d_train;							//学習データ
	std::vector<thrust::device_vector<float>> d_trains;						//trains[エポック数][入力データ数](データの次元数)
	thrust::device_vector<float> d_sdev;							//標準偏差σ1(d_sdev[0]),σ2(d_sdev[1])
	thrust::device_vector<float> d_rot1, d_rot2;					//第一/第二rotation
	thrust::device_vector<float> d_aveVec;						//平均ベクトル
	thrust::device_vector<float> d_umat;							
	/*--- GPU用計算変数ここまで ---*/

	/*--- CPU用計算変数ここから ---*/
	thrust::host_vector<float> h_mapWeight;						//J×I行列のマップの領域確保に利用
	thrust::host_vector<float> h_weightS;						//代表ベクトルWij 
	thrust::device_vector<float> h_cntWeightS;					//代表ベクトルの分類数
	thrust::host_vector<float> h_node;							//node
	thrust::host_vector<int> h_bmuPos;							//d_bmuPos[0]=x,d_bmuPos[1]=y
	//thrust::host_vector<float> h_train;							//学習データ
	thrust::host_vector<float> h_trains;						//trains[エポック数][入力データ数](データの次元数)
	thrust::host_vector<float> h_sdev;							//標準偏差σ1(d_sdev[0]),σ2(d_sdev[1])
	thrust::host_vector<float> h_rot1, h_rot2;					//第一/第二rotation
	thrust::host_vector<float> h_aveVec;							//平均ベクトル
	/*--- CPU用計算変数ここまで ---*/
	
	int getBMUIndex();		//
	void setBMUPosition();	//


	void BMU(float* input_xk);
	void CalcWeightS(float* input_xk, int Lnum);
	void UpdateMapWeight(int Lnum);

	/*--- GPU利用関数 ---*/
	void InitMapWeightRand();//ランダムに初期化を行う
	void InitMapWeightBatch();
	void searchBMUFromGPU(int epoc_num,int data_size);		//epoc_num * data_size + vec_dimで座標を決定

	void d_showWeightS();
	void d_showMapWeight();

public:
	BLSOM(int vec_dim, int map_width);
	BLSOM(int vec_dim, int map_width, int map_height);
	BLSOM(int vec_dim, int map_width, int map_height, int device);
	BLSOM(int vec_dim, int map_width, int map_height, int device,int gpuFlag);
	~BLSOM();
	void Init(const float sdev1, const float sdev2, const float* rot1, const float* rot2, const float *aveVec);
	void InitMapWeight(int mode=InitBatch);	//初期マップの作成
	void Learning(int Lnum);
	void Classification();	///Learning実行後、エポック事の最近傍ノードを割り振る

	float* GetSOMMap();

	/*---Function to specify hyper parameter ---*/
	void SetHyperParameter(double initAlfa, double initBeta, double timeAlfa, double timeBeta);		///initAlfa=学習係数 initBeta学習=半径 timeAlfa=時定数 timeBeta=時定数
	void SetStandardDeviation(float sdev1, float sdev2);
	void SetRotation(float* rot1, float *rot2);
	void SetAverageVecter(float *aveVec);

	/*---　Function to load training data　---*/
	void SetTrainingData(const std::vector<std::vector<float>> train);
	void SetTrainingData(const std::vector<std::vector<std::vector<float>>> train);

	void Test();

	/*--- getter ---*/
	int MapHeight() {
		return this->map_height;
	}
	int MapWidth() {
		return this->map_width;
	}
	int VectorDim() {
		return this->vec_dim;
	}

	/*--- setter ---*/
	void SetMapHeight(int height) {
		this->map_height = height;
	}
	void SetMapWidth(int width) {
		this->map_width = width;
	}
	void SetVecDim(int vecDim) {
		this->vec_dim = vecDim;
	}
	std::vector<std::vector<std::vector<double> > > GetMapWeight();
	std::vector<std::vector<double> >GetUMatrix();


	void check_mapWeight();
};