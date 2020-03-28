
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
using namespace std;


//take the file path name,expected row,col number. Return the pointer of the float array
float* readCSV(string pathName, int targetRow, int targetCol) {
	float* extraContent = new float[targetRow * targetCol - 1];
	ifstream data(pathName);
	string line;
	vector<std::vector<std::string> > parsedCsv;

	int count = 0;
	while (getline(data, line))
	{
		stringstream ss(line);
		for (float i; ss >> i;) {

			extraContent[count] = i;
			count += 1;
			if (ss.peek() == ',')
				ss.ignore();
		}

	}
	return extraContent;
}