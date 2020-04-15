#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <sys/time.h>
#include <unistd.h>

// time stamp function in seconds 
double getTimeStamp() {     
    struct timeval  tv ;     
    gettimeofday( &tv, NULL ) ;    
    return (double) tv.tv_usec/1000000 + tv.tv_sec ; 
} 

float ** training_x; //10000*1024 --> 2556*1024
float ** training_y; //10000*1 --> 2556*1
float ** testing_x; //504*1024
float ** testing_y; //504*1


float * weight; //1024 * 1
float * weight_b;
float * predicting_z; //2556 * 1
float * softmax_a; //2556 * 1

//////////////////////////////////////////////
float * softmax_a1; //2556 * 1
//////////////////////////////////////////////

float * dz; //2556 * 1 or 504*1
float * dz1; //2556 * 1 or 504*1
float * grad; //1024 * 1
float * grad1; //1024 * 1
float * grad_b; //1024 * 1

float * total_loss;

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
    training_x = (float**)malloc(sizeof(float*) * 3500);
    for(int i = 0; i < 3500; i++){
        training_x[i] = (float*)malloc(sizeof(float) * 784);
    }

    training_y = (float**)malloc(sizeof(float*) * 3500);
    for(int i = 0; i < 3500; i++){
        training_y[i] = (float*)malloc(sizeof(float) * 1);
    }

    testing_x = (float **)malloc(sizeof(float*) * 145);
    for(int i = 0; i < 145; i++){
        testing_x[i] = (float*)malloc(sizeof(float) * 784);
    }

    testing_y = (float **)malloc(sizeof(float*) * 145);
    for(int i = 0; i < 145; i++){
        testing_y[i] = (float*)malloc(sizeof(float) * 1);
    }
}

void malloc_weight(void){
    weight = (float*)malloc(sizeof(float*) * 784);
    weight_b = (float*)malloc(sizeof(float*) * 1);
}

void initialize_weight(){
    for (int j = 0; j < 784; j++) {
        weight[j] = 0;
        // printf("weight %f\n", weight[i][j]);
    }
    weight_b[0] = 0;
}

void malloc_z(int size){
    predicting_z = (float*)malloc(sizeof(float*) * size);
}

void initialize_z(int size){
    for (int i = 0; i < size; i++) {
        predicting_z[i] = 0;
    }
}

void dot_mul(float** data, int size){
    initialize_z(size);

    for (int i = 0; i < size; i++) {
        for (int k = 0; k < 784; k++) {
            predicting_z[i] += weight[k]*data[i][k];
        }
        predicting_z[i] += weight_b[0];
    }
    
    // printf("predicting_z %f\n", predicting_z[1]);
    // for (int i = 0; i < size; i++) printf("predicting_z %f\n", predicting_z[i]);
}

void malloc_a(int size){
    softmax_a = (float*)malloc(sizeof(float*) * size);
    softmax_a1 = (float*)malloc(sizeof(float*) * size);
}

void initialize_a(int size){
    for (int i = 0; i < size; i++) {
        softmax_a[i] = 0;
        softmax_a1[i] = 0;
    }
}


//////////////////////////////////////////////////////////////////////
void sigmoid(int size){
    for (int i = 0; i < size; i++){
        if(predicting_z[i] > 10) {
            softmax_a[i] = 1;
        }
        else if(predicting_z[i]< -10) {
            softmax_a[i] = 0;
        }
        else {
            softmax_a[i] = 1 / (1 +(exp(-1 * (double)predicting_z[i])));
        }
    }
    // printf("softmax_a %f\n", softmax_a[3499]);
    // for (int i = 0; i < size; i++) printf("softmax_a %f\n", softmax_a[i]);
}

void sigmoid_minus(int size){
    for (int i = 0; i < size; i++){
        if(predicting_z[i] > 10) {
            softmax_a1[i]= 0;
        }
        else if(predicting_z[i] < -10) {
            softmax_a1[i] = 1;
        }
        else {
            softmax_a1[i] = 1 / (1 +(exp((double)predicting_z[i])));
        }
    }
    // for (int i = 0; i < size; i++) printf("softmax_a1 %f\n", softmax_a1[i]);
}
//////////////////////////////////////////////////////////////////////

void malloc_dz(int size){
    dz = (float*)malloc(sizeof(float*) * size);
    dz1 = (float*)malloc(sizeof(float*) * size);
}

void initialize_dz(int size){
    for (int i = 0; i < size; i++) {
            dz[i] = 0;
            dz1[i] = 0;
    }
}

void malloc_grad(void){
    grad = (float*)malloc(sizeof(float*) * 784);
    grad1 = (float*)malloc(sizeof(float*) * 784);
    grad_b = (float*)malloc(sizeof(float*) * 1);
}

void initialize_grad(void){
    for (int j = 0; j < 784; j++) {
        grad[j] = 0;
        grad1[j] = 0;
    }
    grad_b[0] = 0;
}
////////////////////////////////loss function////////////////////////////
void malloc_loss(void){
    total_loss = (float*)malloc(sizeof(float*) * 1);
}

void initialize_loss(void){
    total_loss[0] = 0;
}

void loss(int size){
    initialize_loss();
    for(int i = 0; i < size; i++){
        total_loss[0] += ((-1 * predicting_z[i] * training_y[i][0]) + log(1 + exp(predicting_z[i])));
    }
    //printf("total loss : %f\n", total_loss[0]);
}


void gradient_descent(float learning_rate) {
    ///////////////////////Linear Regression//////////////////////

    // initialize_dz(3500);
    // for (int i = 0; i < 3500; i++) {
    //     dz[i] = predicting_z[i] - training_y[i][0];
    // }
    // printf("dz %f\n", dz[0]);

    // initialize_grad();
    // for (int i = 0; i < 784; i++){
    //     for (int j = 0; j < 3500; j++){
    //         grad[i] += training_x[j][i] * dz[j];
    //     }
    //     grad[i] = grad[i] / 3500;
    // }

    // // Update weight
    // for (int i = 0; i < 784; i++) {
    //     weight[i] -= (learning_rate * grad[i]);
    // }

    // // update weight_b
    // for (int i = 0; i < 3500; i++){
    //     grad_b[0] += dz[i];
    // }
    // grad_b[0] /= 3500;

    // weight_b[0] -= (learning_rate * grad_b[0]);

    //////////////////////////////////Logistic Regression////////////////////////////////////
    initialize_dz(3500);
    for (int i = 0; i < 3500; i++){
        dz[i] = (softmax_a[i] - training_y[i][0]);
        // if(training_y[i][0] == 0){
        //     dz[i] = softmax_a[i];
        // }
        // else{
        //     dz1[i] = -1 * softmax_a1[i];//Front item
        // }
    }
    // for(int i = 0; i < 3500; i++) printf("dz %f\n", dz[i]);

    initialize_grad();
    //////////////// Update weight
    for (int i = 0; i < 784; i++){
        for (int j = 0; j < 3500; j++){
            // grad[i] +=  learning_rate * training_x[j][i] * dz[j];
            // grad1[i] +=  learning_rate * training_x[j][i] * dz1[j];
            grad[i] +=  training_x[j][i] * dz[j];
            // grad1[i] +=  training_x[j][i] * dz1[j];
        }
        grad[i] = (grad[i]) / 3500;
    }


    for (int i = 0; i < 784; i++) {
        weight[i] -= (learning_rate * grad[i]);
        //weight[i] -= (grad[i]);
    }
    //for(int i = 0; i < 784; i++) printf("weight %f\n", weight[i]);


}

void testing() {
    dot_mul(training_x, 3500);
    sigmoid(3500);

    int accuracy = 0;

    for (int i = 0; i < 3500; i++) {
        if((training_y[i][0]) == 1){
            if(softmax_a[i] > 0.5){
                accuracy++;
            }
        } 
        else{
            if(softmax_a[i] < 0.5){
                accuracy++;
            }
        }
    }

    // printf("accuracy : %f\n", accuracy/3500.0);
    // return accuracy/3500.0;
}

void logistic_regression(int iter, float learning_rate) {
    for (int i = 0; i < iter; i++) {
        initialize_z(3500);
        dot_mul(training_x, 3500);
        sigmoid(3500);
        sigmoid_minus(3500);
        gradient_descent(learning_rate);
        loss(3500);
        testing();
        // printf("Iteration : %d\n", i);
    }
}

// void testing() {
//     dot_mul(training_x, 3500);
//     sigmoid(3500);

//     int accuracy = 0;

//     for (int i = 0; i < 3500; i++) {
//         if((training_y[i][0]) == 1){
//             if(softmax_a[i] > 0.5){
//                 accuracy++;
//             }
//         } 
//         else{
//             if(softmax_a[i] < 0.5){
//                 accuracy++;
//             }
//         }
//     }

//     printf("accuracy : %f\n", accuracy/3500.0);
//     // return accuracy/3500.0;
// }

int main(){
    malloc_host();
    readCSV("train_x.csv", training_x, 3500, 784);
    readCSV("train_y.csv", training_y, 3500, 1);
    readCSV("test_x.csv", testing_x, 145, 784);
    readCSV("test_y.csv", testing_y, 145, 1);
    printf("label %f\n", training_y[122][0]);

    malloc_weight();
    initialize_weight();

    malloc_z(3500);
    malloc_a(3500);
    malloc_dz(3500);
    malloc_grad();
    malloc_loss();

    int iter = 10000;
    float learning_rate = 0.01;

    double timeStamp1 = getTimeStamp() ;

    logistic_regression(iter, learning_rate);

    double timeStamp2 = getTimeStamp() ;

    printf("%.6f\n", timeStamp2-timeStamp1);
    
    // free(predicting_z);
    // free(softmax_a);
    // free(dz);
    // free(grad);

    // for(int i = 0; i < 784; i++) printf("weight %f\n", weight[i]);

    malloc_z(3500);
    malloc_a(3500);
    //float test_accuracy = testing();

    // printf("The testing accuracy is %f\n", test_accuracy);
    

    // free(predicting_z);
    // free(softmax_a);

    // free(training_x);
    // free(training_y);
    // free(testing_x);
    // free(testing_y);

    // free(weight);
}


