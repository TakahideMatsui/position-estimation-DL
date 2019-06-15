#pragma once
#include<string>
#include<vector>
#include<fstream>
#include<iostream>

std::vector<float> LoadStandardDev(std::string fileName, char delim = ' ', bool header = true);

std::vector<std::vector<float>> LoadRotation(std::string fileName, char delim = ' ', bool header = true);

std::vector<float> LoadTrain(std::string fileName, char delim = ' ', bool header = true);

std::vector<std::vector<float>> LoadTrains(std::string fileName, char delim, bool header=true);

std::vector<float> LoadAverageVector(std::string fileName, char delim = ' ', bool header = true);