#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

using namespace std;

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define NUM_THREADS 1024
cudaStream_t stream;

int features = 1024;
int sampels = 10000;
int classes = 10;

float ** training_x; //3500 * 784
float ** training_y; //3500 * 1
float ** testing_x; //145 * 784
float ** testing_y; //145 * 1

float ** label_onehot; //3500 * 10

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
    for(int i = 0; i < 2000; i++){
        testing_x[i] = (float*)malloc(sizeof(float) * 1024);
    }

    testing_y = (float **)malloc(sizeof(float*) * 2000);
    for(int i = 0; i < 2000; i++){
        testing_y[i] = (float*)malloc(sizeof(float) * 1);
    }

    label_onehot = (float **)malloc(sizeof(float*) * 10000);
    for (int i = 0; i < 10000; i++)
    {
        label_onehot[i] = (float*)malloc(sizeof(float) * 10);
    }
}

__global__ void Mult_GPU( float *a,  float *b, float *result,  const int M, const int N, const int S) // M should be batch size
{
    int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < M * S)
    {
        int row = threadId / S;
        int column = threadId % S;

        result[threadId] = 0;
        for (int i = 0; i < N; i++)
        {
            result[threadId] += a[row * N + i] * b[i * S + column];
        }
    }
}

__global__ void softmax_sum( float *predict, float *sum, const int label_size, const int data_size ){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data_size){
        for(int i = 0; i < label_size; i++){
            sum[tid] += exp(predict[tid * label_size + i]);
        }
    }
}

__global__ void max( float *predict, float *max, const int label_size, const int data_size ){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < data_size){
        for(int i = 0; i < label_size; i++){
            int max_index = 0;
            max[tid] =  predict[tid * label_size];
            if(predict[tid * label_size + max_index] <  predict[tid * label_size + i]){
                max[tid] = predict[tid * label_size + i];
            }
        }
    }
}

__global__ void normalize(float *predict, float *max, const int label_size, const int data_size){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data_size){
        for(int i = 0; i < label_size; i++){
            predict[tid * label_size + i] -= max[tid];
        }
    }
}

__global__ void softmax( float *softmax_value, float *predict, float *sum,const int label_size, const int data_size ){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data_size){
        for(int i = 0; i < label_size; i++){
            softmax_value[tid * label_size + i] =  exp(predict[tid * label_size + i]) / sum[tid];
        }
    }
}

__global__ void dz(float *softmax_value, float *label, float *dz, const int label_size, const int data_size){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
              + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data_size){
        for(int i = 0; i < label_size; i++){
            dz[tid * label_size + i] = softmax_value[tid * label_size + i] - label[tid * label_size + i];
        }
    }
}

__global__ void grad(float *train_data, float *dz, float *grad, const int label_size, const int data_size, const int weight_size){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
              + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < weight_size){
        for(int i = 0; i < label_size; i++){
            for(int j = 0; j < data_size; j++)
            grad[tid * label_size + i] += train_data[j * weight_size + tid] * grad[j * label_size + i];
        }
    }
}

__global__ void weight_update(float *weight, float *grad, const int label_size, const int weight_size, const float learning_rate){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
              + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < weight_size){
        for(int i = 0; i < label_size; i++){
            grad[tid * label_size + i] /= 10000;
            weight[tid * label_size + i] -= (learning_rate * grad[tid * label_size + i]);
        }
    }
}

__global__ void initialize_dz(float *dz, const int label_size, const int data_size){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
              + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data_size){
        for(int i = 0; i < label_size; i++){
            dz[tid * label_size + i] = 0;
        }
    }
}

__global__ void initialize_grad(float *grad, const int label_size, const int weight_size){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
              + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < weight_size){
        for(int i = 0; i < label_size; i++){
            grad[tid * label_size + i] = 0;
        }
    }
}

__global__ void initialize(float *sum, float *predict, const int data_size, const int label_size){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
              + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data_size){
        sum[tid] = 0;
        for(int i = 0; i < label_size; i++){
            predict[tid * label_size + i] = 0;
        }
    }

}



int main(){
    // malloc_host();
    malloc_host();
    readCSV("training_x.csv", training_x, 10000,1024);
    readCSV("training_y.csv", training_y, 1024, 1);
    readCSV("testing_x.csv", testing_x, 2000, 1024);
    readCSV("testing_y.csv", testing_y, 2000, 1);
    printf("label %f\n", training_y[9998][0]);

    float learning_rate = 0.01;
    int iter = 1;

    int data_size = 10000;
    int label_size = 10;
    int weight_size = 1024;

    int tratin_data_bytes = 10000 * 1024 * sizeof(float);
    int weight_bytes = 1024 * 10 * sizeof(float);
    int predict_bytes = 10000 * 10 * sizeof(float);

    float *h_train_data = (float *) malloc( tratin_data_bytes ) ;
    float *h_train_data_T = (float *) malloc( tratin_data_bytes ) ;
    float *h_label_onehot = (float *) malloc( predict_bytes ) ;
    float *h_weight = (float *) malloc( weight_bytes ) ;
    float *h_predict = (float *) malloc( predict_bytes ) ;
    float *h_max = (float *) malloc( 10000 * sizeof(float) ) ;
    float *h_sum = (float *) malloc( 10000 * sizeof(float) ) ;
    float *h_softmax = (float *) malloc( predict_bytes ) ;
    float *h_dz = (float *) malloc( predict_bytes ) ;
    float *h_grad = (float *) malloc( weight_bytes ) ;


    ////////////////////// Initialize //////////////////////
    ////////////////////// One Hot //////////////////////
    for(int i = 0; i < data_size; i++){
        for(int j = 0; j < weight_size; j++){
            h_train_data_T[j * 10000 + i] = training_x[i][j];
        }
    }

    for(int i = 0; i < data_size; i++){
        label_onehot[i][(int(training_y[i][0] - 1))] = 1;
        if(i == 1){
            printf("training_y : %f\n", training_y[1][0]);
            for(int j = 0; j < 10; j++) printf("onehot : %f\n", label_onehot[i][j]);
        }
    }

    for(int i = 0; i < data_size; i++){
        for(int j = 0; j < label_size; j++){
            h_label_onehot[i * label_size + j] = label_onehot[i][j];
        }
    }

    for(int i = 0; i < data_size; i++){
        for(int j = 0; j < weight_size; j++){
            h_train_data[i * weight_size + j] = training_x[i][j];
        }
    }

    for(int i = 0; i < weight_size; i++){
        for(int j = 0; j < label_size; j++){
            h_weight[i * label_size + j] = 1 ;
        }
    }
    //////////////////// Initialize //////////////////////


    ///////////////////////////////// GPU_SIDE ///////////////////////////////////
    float *d_train_data, *d_train_data_T, *d_label, * d_weight, *d_predict, *d_predict_sum, *d_sum, *d_max, *d_softmax_value;
    float *d_dz, *d_grad;

    cudaGetErrorString(cudaMalloc( (void **) &d_train_data, tratin_data_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_train_data_T, tratin_data_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_label, predict_bytes)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_predict, predict_bytes)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_weight, weight_bytes)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_sum, 10000 * sizeof(float))) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_softmax_value, predict_bytes)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_dz, predict_bytes)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_grad, weight_bytes)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_max, 10000 * sizeof(float))) ;

    // //Configure blockDim
    int bdx = 32, bdy = 32;
    while(data_size > bdx * 65535)
    {
        bdx = bdx * 2;
        bdy = bdy / 2;
    }
    while(weight_size > bdy * 65535)
    {
        bdy = bdy * 2;
        bdx = bdx / 2;
    }
    dim3 blockDim( bdx,bdy ) ; // you will want to configure this
    dim3 gridDim( (int)((data_size + blockDim.x-1)/blockDim.x), (int)((weight_size + blockDim.y-1)/blockDim.y) ) ;

    //////////////////////////////// invoke Kernel (Logistic Regression) ////////////////////////////////
    
    cudaGetErrorString(cudaMemcpy( d_train_data_T, h_train_data_T, tratin_data_bytes, cudaMemcpyHostToDevice )) ;
    cudaGetErrorString(cudaMemcpy( d_train_data, h_train_data, tratin_data_bytes, cudaMemcpyHostToDevice )) ;
    cudaGetErrorString(cudaMemcpy( d_weight, h_weight, weight_bytes, cudaMemcpyHostToDevice )) ;
    cudaGetErrorString(cudaMemcpy( d_label, h_label_onehot, predict_bytes, cudaMemcpyHostToDevice )) ;

    for(int train  = 0; train < 10; train++){
        //Initialize
        initialize<<<gridDim, blockDim>>>(d_sum, d_predict, data_size, label_size);
        cudaGetErrorString(cudaDeviceSynchronize());
        initialize_dz<<<gridDim, blockDim>>>(d_dz, label_size, data_size);
        cudaGetErrorString(cudaDeviceSynchronize());
        initialize_grad<<<gridDim, blockDim>>>(d_grad, label_size, weight_size);
        cudaGetErrorString(cudaDeviceSynchronize());

        // DOT
        Mult_GPU<<<gridDim, blockDim>>>( d_train_data, d_weight, d_predict, data_size, weight_size, label_size) ;
        cudaGetErrorString(cudaDeviceSynchronize());	

        max<<<gridDim, blockDim>>>( d_predict, d_max, label_size, data_size );
        cudaGetErrorString(cudaDeviceSynchronize());	
        normalize<<<gridDim, blockDim>>>(d_predict, d_max, label_size, data_size);
        cudaGetErrorString(cudaDeviceSynchronize());
        
        // Softmax

        softmax_sum<<<gridDim, blockDim>>>( d_predict, d_sum, label_size, data_size );
        cudaGetErrorString(cudaDeviceSynchronize());
        softmax<<<gridDim, blockDim>>>( d_softmax_value, d_predict, d_sum, label_size, data_size );
        cudaGetErrorString(cudaDeviceSynchronize());

        // Weight Update
        dz<<<gridDim, blockDim>>>(d_softmax_value, d_label, d_dz, label_size, data_size);
        cudaGetErrorString(cudaDeviceSynchronize());
        Mult_GPU<<<gridDim, blockDim>>>( d_train_data_T, d_dz, d_grad, weight_size, data_size, label_size) ;
        cudaGetErrorString(cudaDeviceSynchronize());
        weight_update<<<gridDim, blockDim>>>(d_weight, d_grad, label_size, weight_size, learning_rate);
        cudaGetErrorString(cudaDeviceSynchronize());
    }

    /////////////////////// Test //////////////////////////
    cudaGetErrorString(cudaMemcpy( h_predict, d_predict, predict_bytes, cudaMemcpyDeviceToHost )) ;
    
    cudaGetErrorString(cudaMemcpy( h_softmax, d_softmax_value, predict_bytes, cudaMemcpyDeviceToHost )) ;

    cudaGetErrorString(cudaMemcpy( h_weight, d_weight, weight_bytes, cudaMemcpyDeviceToHost )) ;

    cudaGetErrorString(cudaMemcpy( h_max, d_max, 10000 * sizeof(float), cudaMemcpyDeviceToHost )) ;

    // int count = 0;
    // for(int i = 0; i < data_size; i++){
    //     if(h_sum[i] == 10.0) count++;
    //     printf("max : %f\n", h_sum[i]);
    // }
    // printf("count 10 num =  %d\n", count);

    for(int i = 0; i < weight_size * label_size; i++){
        printf("dz: %f\n", h_weight[i]);
    }
    // for(int i = 0; i < 10000; i++){
    //     printf("max: %f\n", log(h_max[i]) );
    // }

    // for(int i = 0; i < data_size * label_size; i++){
    //     // printf("softmax: %f\n", log(h_sum[i / label_size]) );
    //     printf("softmax: %f\n", h_softmax[i] );
    // }
    

    // float total_error = 0;
    // for(int i = 0; i < data_size; i++){
    //     for(int j = 0; j < label_size; j++){
    //         total_error -= label_onehot[i][j] * log(h_softmax[i * label_size + j]) ;
    //     }
    // }
    // printf("error: %f\n", total_error );

}