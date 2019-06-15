#include<iostream>
#include"BLSOM.h"
#include"SelectGPU.h"
#include"LoadDataSet.h"
#include<curand_kernel.h>
#include<algorithm>

#define MAP_WIDTH 200
#define MAP_HEIGHT 50
#define TRAIN_NUM 200
#define EPOC_NUM 0

int WriteSOMMAP(std::string fileName, float* map, int map_vec, int map_width, int map_height) {
	std::ofstream ofs;
	ofs.open(fileName, 'w');

	if (!ofs) {
		std::cerr << "can't opne file" << std::endl;
		return EXIT_FAILURE;
	}

	ofs << map_vec << std::endl;
	ofs << map_width << std::endl;
	ofs << map_height << std::endl;

	for (int i = 1; i < map_height*map_width; i++) {
		for (int v = 0; v < map_vec; v++) {
			ofs << *map << " ";
			map++;
		}
		ofs << "\n";
	}
	ofs.close();

	return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
	int device;
	int vec_dim;
	int map_width;
	int map_height;
	float* som;
	std::shared_ptr<float> map_weight;
	std::vector<std::vector<float>> train;
	std::vector<std::vector<std::vector<float>>> epocs;


	std::vector<float> ave_vec;
	std::vector<std::vector<float>> rotation;
	std::vector<float> sdev;


	/* load init data */
	train = LoadTrains("sample\\No1.txt", '\t');
	ave_vec = LoadAverageVector("sample\\vector_Ave.txt");
	rotation = LoadRotation("sample\\rotation.txt");
	sdev = LoadStandardDev("sample\\sdev.txt");

	map_width = MAP_WIDTH;
	map_height = MAP_HEIGHT;
	vec_dim = ave_vec.size();

	/* Create BLSOM Object */
	BLSOM test = BLSOM(vec_dim, map_width);
	
	/* Set Learning Data */
	test.Init(sdev[0], sdev[1], rotation[0].data(), rotation[1].data(), ave_vec.data());
	test.SetTrainingData(train);

	/* Chose Method of initialization */
	test.InitMapWeight(INIT_BATCH);

	/* Get initial map */
	som = test.GetSOMMap();
	WriteSOMMAP("C:\\Users\\Kai\\Desktop\\mori_PCA\\init_batch_map.txt", som, vec_dim, map_width, test.MapHeight());

	/* Learning */
	test.Learning(50);
	
	/* Get Learned Map */
	som = test.GetSOMMap();
	WriteSOMMAP("C:\\Users\\Kai\\Desktop\\mori_PCA\\result_random_20190406.txt", som, vec_dim, map_width, test.MapHeight());
	
	return 0;
}