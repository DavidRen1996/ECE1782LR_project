#include <stdio.h>
#include <stdlib.h>
#include <string.h>


float ** training_x;
float ** training_y;
float ** testing_x;
float ** testing_y;

void getData(float * res, char buff[])
{
    char *token = strtok(buff," ,");
    int counter=0;
    
    while( token != NULL ) 
    {
        counter++; 
        res[counter-1] = atof(token);
        token = strtok(NULL," ,");
    }	  
}

void readCSV(char* file, float** mat, int x_dim, int y_dim)
{
    FILE* stream = fopen(file, "r");
    int size_per_pic = y_dim * 30;
    char line[size_per_pic];
    int num;
    if (stream == NULL) {
        perror ("Error opening file");
        return;
    }

    int i = 0;
    while (fgets(line, size_per_pic, stream))
    {
        char* tmp = strdup(line);
        getData(mat[i], tmp);
        i++;
    }
}

void malloc_host(void){
    training_x = (float**)malloc(sizeof(float*) * 10000);
    for(int i = 0; i < 10000; i++){
        training_x[i] = (float*)malloc(sizeof(float) * 1024);
    }

    training_y = (float**)malloc(sizeof(float*) * 10000);
    for(int i = 0; i < 10000; i++){
        training_y[i] = (float*)malloc(sizeof(float) * 1);
    }

    testing_x = (float **)malloc(sizeof(float*) * 2000);
    for(int i = 0; i < 10000; i++){
        testing_x[i] = (float*)malloc(sizeof(float) * 1024);
    }

    testing_y = (float **)malloc(sizeof(float*) * 2000);
    for(int i = 0; i < 10000; i++){
        testing_y[i] = (float*)malloc(sizeof(float) * 1);
    }
}

int main(){
    malloc_host();
    readCSV("training_x.csv", training_x, 10000, 1024);
    readCSV("training_y.csv", training_y, 10000, 1);
    readCSV("testing_x.csv", testing_x, 2000, 1024);
    readCSV("testing_y.csv", testing_y, 2000, 1);
}