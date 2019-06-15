#include"LoadDataSet.h"


std::vector <std::string> split(std::string str, char delim) {
	std::vector<std::string> elements;
	std::string item;

	for each (char ch in str)
	{
		if (ch == delim) {
			if (!item.empty()) {
				elements.push_back(item);
			}
			item.clear();
		}
		else {
			item += ch;
		}
	}
	if (!item.empty()) {
		elements.push_back(item);
	}
	return elements;
}


std::vector<float> LoadStandardDev(std::string fileName, char delim, bool header) {
	std::vector<float> sdev;
	std::vector<std::string> elem;
	std::ifstream ifs;
	std::string line;

	ifs.exceptions(std::ifstream::badbit);
	try {
		ifs.open(fileName);

		if (!ifs) {
			std::cerr << "File opening failed" << std::endl;
			exit(1);
		}

		if (header) {
			getline(ifs, line);
		}

		while (getline(ifs, line))
		{
			if (!line.empty()) {
				elem = split(line, delim);
				sdev.push_back(stof(elem[1]));
			}
		}
		ifs.close();
	}
	catch(std::ifstream::failure e) {
		std::cerr <<"Exception opening/reading/closing file" << std::endl;
	}
	return sdev;
}

std::vector<std::vector<float>> LoadRotation(std::string fileName, char delim, bool header) {
	std::vector<std::vector<float>> rotation;
	std::vector<float> rot1;
	std::vector<float> rot2;

	std::vector<std::string> elem;
	std::ifstream ifs;
	std::string line;

	ifs.exceptions(std::ifstream::badbit);
	try {
		ifs.open(fileName);

		if (!ifs) {
			std::cerr << "File opening failed" << std::endl;
			exit(1);
		}

		if (header) {
			getline(ifs, line);
		}

		while (getline(ifs, line))
		{
			if (!line.empty()) {
				elem = split(line, delim);
				rot1.push_back(stof(elem[1]));
				rot2.push_back(stof(elem[2]));
			}
		}
		ifs.close();
	}
	catch (std::ifstream::failure e) {
		std::cerr << "Exception opening/reading/closing file" << std::endl;
	}

	rotation.push_back(rot1);
	rotation.push_back(rot2);
	return rotation;
}

std::vector<float> LoadTrain(std::string fileName, char delim, bool header) {
	std::vector<float> train;
	std::vector<std::string> elem;
	std::ifstream ifs;
	std::string line;

	ifs.exceptions(std::ifstream::badbit);
	try {
		ifs.open(fileName);
		
		if (!ifs) {
			std::cerr << "File opening failed" << std::endl;
			exit(1);
		}

		if (header) {
			getline(ifs, line);
		}

		while (getline(ifs, line))
		{	
			elem = split(line, delim);
			for each(std::string el in elem)
				train.push_back(stof(el));
		}
		ifs.close();
	}
	catch (std::ifstream::failure e) {
		std::cerr << "Exception opening/reading/closing file" << std::endl;
	}
	return train;
}

std::vector<std::vector<float>> LoadTrains(std::string fileName, char delim, bool header) {
	std::vector<std::vector<float>> trains;
	std::vector<float> train;
	std::vector<std::string> elem;
	std::ifstream ifs;
	std::string line;

	ifs.exceptions(std::ifstream::badbit);
	try {
		ifs.open(fileName);

		if (!ifs) {
			std::cerr << "File opening failed" << std::endl;
			exit(1);
		}

		if (header) {
			getline(ifs, line);
		}

		while (getline(ifs, line))
		{
			elem = split(line, delim);
			for each(std::string el in elem)
				train.push_back(stof(el));
			trains.push_back(train);
			train.clear();
		}
		ifs.close();
	}
	catch (std::ifstream::failure e) {
		std::cerr << "Exception opening/reading/closing file" << std::endl;
	}
	return trains;
}

std::vector<float> LoadAverageVector(std::string fileName, char delim, bool header) {
	std::vector<float> ave;
	std::vector<std::string> elem;
	std::ifstream ifs;
	std::string line;

	ifs.exceptions(std::ifstream::badbit);
	try {
		ifs.open(fileName);

		if (!ifs) {
			std::cerr << "File opening failed" << std::endl;
			exit(1);
		}

		if (header) {
			std::getline(ifs, line);
		}

		while (std::getline(ifs, line))
		{
			if (!line.empty()) {
				elem = split(line, delim);
				ave.push_back(stof(elem[1]));
			}
		}
		ifs.close();
	}
	catch (std::ifstream::failure e) {
		std::cerr << "Exception opening/reading/closing file" << std::endl;
	}
	return ave;
}
