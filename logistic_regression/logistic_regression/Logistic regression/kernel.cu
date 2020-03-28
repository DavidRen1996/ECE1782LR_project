
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "readStream.h"
using namespace std;


//take the file path name,expected row,col number. Return the pointer of the float array


int main()
{
    
	float *train_X=readCSV("D:\\Microsoft\\CUDA_projects\\training_x.csv", 10000, 1024);
	float* train_Y = readCSV("D:\\Microsoft\\CUDA_projects\\training_y.csv", 10000, 1);
	float* test_X = readCSV("D:\\Microsoft\\CUDA_projects\\testing_x.csv", 2000, 1024);
	float* test_Y = readCSV("D:\\Microsoft\\CUDA_projects\\testing_y.csv", 2000, 1);
	cout << test_X[0] << " "<< test_X[2000 * 1 - 1];//hard coded row and col number
    return 0;
}

