#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

using namespace std;

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define NUM_THREADS 1024
cudaStream_t stream;

float ** training_x; //10000*1024 --> 2556*1024
float ** training_y; //10000*1 --> 2556*1
float ** testing_x; //504*1024
float ** testing_y; //504*1

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

// void malloc_weight(void){
//     weight = (float*)malloc(sizeof(float*) * 1024);
    
// }

void initialize_weight(float* weight){
    for (int j = 0; j < 1024; j++) {
        weight[j] = 0;
        // printf("weight %f\n", weight[i][j]);
    }
}

// train_data size a[M][N]    M = data_size; N = 784;
// weight size b[N][S]        N = 784; S = 10
// result size result[M][S]   M = data_size; S = 10

void Mult_CPU( float * a, float * b, float *result, float *weight_b,const int M,const int N,const int S) // M should be batch size
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < S; j++)
        {
            int index = i * S + j;
            result[index] = 0;

            //循环计算每一个元素的结果
            for (int k = 0; k < N; k++)
            {
                result[index] += a[i * N + k] * b[k * S + j];
            }
            result[index] += weight_b[0];
        }
    }
}
//optimization ideas:1.replace global memory with shared memory 2.transpose the input array to ensure coalesced access 3.avoid share mem conflict
//optimization that won't work:squeeze threadId into smaller ones
__global__ void Mult_GPU( float *a,  float *b, float *result,  float *weight_b, const int M, const int N, const int S) // M should be batch size
{
    int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
	//__shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
	//__shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];
    if (threadId < M * S)
    {
        int row = threadId / S;
        int column = threadId % S;

        result[threadId] = 0;
        for (int i = 0; i < N; i++)
        {
            result[threadId] += a[row * N + i] * b[i * S + column];
        }
        result[threadId] += weight_b[0];
    }
}
// no optimization could be done:No reused memory, all threadId will be used, no memory conflict found
__global__ void sigmoid_GPU(float *a, float *result, const int size){
    const int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size){
        if(a[tid] > 10){
            result[tid] = 1;
        }
        else if(a[tid] < -10){
            result[tid] = 0;
        }
        else{
            result[tid] = 1 / ( 1 + exp( (double)(-1 * a[tid])) );
        }
    }
}
// no optimization could be done:No reused memory, all threadId will be used, no memory conflict found
__global__ void sigmoid_minus_GPU(float *a, float *result, const int size){
    const int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size){
        if(a[tid] > 10 ){
            result[tid] = 0;
        }
        else if(a[tid] < -10){
            result[tid] = 1;
        }
        else{
            result[tid] = 1 / ( 1 + exp( (double)(a[tid])) );
        }
    }
}

void sigmoid_CPU(float *a, float *result, const int size){
    for (int i = 0; i < size; i++){
        if(a[i] > 10){
            result[i] = 1;
        }
        else if(a[i] < -10){
            result[i] = 0;
        }
        else{
            result[i] = 1 / ( 1 + exp( (double)(-1 * a[i])) );
        }
    }
}
// no optimization could be done:No reused memory, all threadId will be used, no memory conflict found
__global__ void initialize_dz(float *dz, float *dz1, const int data_size){
    const int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data_size){
        dz[tid] = 0;
        dz1[tid] = 0;
    }
}

// no optimization could be done:No reused memory, all threadId will be used, no memory conflict found
__global__ void dz_GPU(float *y_label, float *y_pre1, float *y_pre2, float *dz, float *dz1, const int data_size){

        const int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                        + blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < data_size){
            if(y_label[tid] == 0){
                dz[tid] = y_pre1[tid];
                dz1[tid] = 0;
            }
            else{
                dz1[tid] = -1 * y_pre2[tid];
                dz[tid] = 0;
            }
        }
}

//optimize solution:1.shared memory 2.not sure if memory access is coalesed
__global__ void grad_GPU(float *train_data, float *dz, float *dz1, float *grad, float *grad1, 
                    const int data_size, const int weight_size, const float learning_rate){

    const int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < weight_size){
        for (int i = 0; i < data_size; i++){
            // grad[tid] +=  learning_rate * train_data[i][tid] * dz[i];
            grad[tid] +=  train_data[i * weight_size + tid] * dz[i];
            grad1[tid] +=  train_data[i * weight_size + tid] * dz1[i];
        }
        grad[tid] = (grad[tid] + grad1[tid]) / data_size ;
    }
}
//cannot optimize
__global__ void grad_b_GPU(float *dz, float *dz1, float *grad_b, const int data_size){
    const int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data_size){
        grad_b[0] += (dz[tid] + dz1[1]) / data_size;
    }
}
//cannot optimize
__global__ void initialize_grad(float *grad, float *grad_b, const int weight_size){
    const int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < weight_size){
        if(tid == 0) grad_b[tid] = 0;
        grad[tid] = 0;
    }
}


//cannot optimize
__global__ void weight_update_GPU(float *weight, float *grad, const int data_size, const int weight_size, const float learning_rate){
    const int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < weight_size){
        weight[tid] -= learning_rate * grad[tid];
    }
}
//cannot optimize
__global__ void weight_b_update_GPU(float * weight_b, float *grad_b, const int data_size, const int weight_b_size, const float learning_rate){
    const int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < weight_b_size){
        // grad_b[tid] /= data_size;
        weight_b[tid] -= learning_rate * grad_b[tid];
    }
}



int main(){
    malloc_host();
    readCSV("train_x.csv", training_x, 3500,784);
    readCSV("train_y.csv", training_y, 784, 1);
    readCSV("test_x.csv", testing_x, 145, 784);
    readCSV("test_y.csv", testing_y, 145, 1);
    printf("label %f\n", training_y[1][0]);

    // for(int i = 0; i < 1024; i++){printf("h_train_data %f \n", training_x[1][i]);}

    //CPU
    float learning_rate = 0.01;

    int data_size = 3500;
    int weight_size = 784;
    int predict_size = 3500;
    int tratin_data_bytes = 3500 * 784 * sizeof(float);
    int weight_bytes = 784 * sizeof(float);
    int predict_bytes = 3500 * sizeof(float);

    float *h_train_data = (float *) malloc( tratin_data_bytes ) ;
    float *h_weight = (float *) malloc( weight_bytes ) ;
    float *h_weight_b = (float *) malloc( 1 * sizeof(float) ) ;
    float *h_label  = (float *) malloc( predict_bytes  ) ; // host result
    float *h_predict  = (float *) malloc( predict_bytes  ) ; // host result
    float *h_softmax = (float *) malloc( predict_bytes ) ;
    float *h_softmax_minus = (float *) malloc( predict_bytes );
    float *h_dz  = (float *) malloc( predict_bytes  ) ;
    float *h_dz1  = (float *) malloc( predict_bytes  ) ;
    float *h_grad = (float *) malloc( weight_bytes ) ;
    float *h_grad1 = (float *) malloc( weight_bytes ) ;
    float *h_grad_b = (float *) malloc( 1 * sizeof(float) ) ;

    // Load initialize data
    for(int i = 0; i < data_size; i++){
        for(int j = 0; j < weight_size; j++){
            h_train_data[i * weight_size + j] = training_x[i][j];
        }
    }

    for(int i = 0; i < weight_size; i++){
        h_weight[i] = 0;
    }
    h_weight_b[0] = 0;

    for(int i = 0; i < data_size; i++){
        h_label[i] = training_y[i][0];
    }

    // // GPU
    float *d_train_data, * d_weight, * d_weight_b, *d_predict, *d_softmax, *d_softmax_minus;
    float *d_label, *d_dz, *d_dz1, *d_grad, *d_grad1, *d_grad_b;

    cudaGetErrorString(cudaMalloc( (void **) &d_train_data, tratin_data_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_label, predict_bytes)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_weight, weight_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_weight_b, 1 * sizeof(float) )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_predict, predict_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_softmax, predict_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_softmax_minus, predict_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_dz, predict_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_dz1, predict_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_grad, weight_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_grad1, weight_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_grad_b,  1 * sizeof(float) )) ;

    cudaGetErrorString(cudaMemcpy( d_train_data, h_train_data, tratin_data_bytes, cudaMemcpyHostToDevice )) ;
    cudaGetErrorString(cudaMemcpy( d_weight, h_weight, weight_bytes, cudaMemcpyHostToDevice )) ;
    cudaGetErrorString(cudaMemcpy( d_weight_b, h_weight_b, 1 * sizeof(float), cudaMemcpyHostToDevice )) ;
    cudaGetErrorString(cudaMemcpy( d_label, h_label, predict_bytes, cudaMemcpyHostToDevice ));


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
 
    for(int train  = 0; train < 1; train++){
        // DOT
        Mult_GPU<<<gridDim, blockDim>>>( d_train_data, d_weight, d_predict, d_weight_b, data_size, weight_size, 1 ) ;
        cudaGetErrorString(cudaDeviceSynchronize()) ;	

        // //Sigmoid
        sigmoid_GPU<<<gridDim, blockDim>>>( d_predict, d_softmax, data_size ) ;
        cudaGetErrorString(cudaDeviceSynchronize()) ;	
        sigmoid_minus_GPU<<<gridDim, blockDim>>>( d_predict, d_softmax_minus, 3500 ) ;
        cudaGetErrorString(cudaDeviceSynchronize()) ;	

        // Update weight (including calculating weight and )
        initialize_dz<<<gridDim, blockDim>>>(d_dz, d_dz1, data_size);
        cudaGetErrorString(cudaDeviceSynchronize());
        initialize_grad<<<gridDim, blockDim>>>(d_grad, d_grad_b,weight_size);
        cudaGetErrorString(cudaDeviceSynchronize());

        dz_GPU<<<gridDim, blockDim>>>(d_label, d_softmax, d_softmax_minus, d_dz, d_dz1, data_size);
        cudaGetErrorString(cudaDeviceSynchronize()) ;	

        grad_GPU<<<gridDim, blockDim>>>( d_train_data, d_dz, d_dz1, d_grad, d_grad1, data_size, weight_size, learning_rate);
        cudaGetErrorString(cudaDeviceSynchronize()) ;	
        weight_update_GPU<<<gridDim, blockDim>>>(d_weight, d_grad, data_size, weight_size, learning_rate);
        cudaGetErrorString(cudaDeviceSynchronize());	

        grad_b_GPU<<<gridDim, blockDim>>>(d_dz, d_dz1, d_grad_b, data_size);
        cudaGetErrorString(cudaDeviceSynchronize());
        weight_b_update_GPU<<<gridDim, blockDim>>>(d_weight_b, d_grad_b, data_size, 1, learning_rate);
        cudaGetErrorString(cudaDeviceSynchronize());

    }

    // cudaGetErrorString(cudaMemcpy( h_grad_b, d_grad_b, 1 * sizeof(float), cudaMemcpyDeviceToHost )) ;
    // // for(int i = 0; i < weight_size; i++) printf("h_weight %f \n", h_weight[i]);
    // printf("h_grad_b %f \n", h_grad_b[0]);
    //////////////////////////////// invoke Kernel (Logistic Regression) ////////////////////////////////
    cudaGetErrorString(cudaMemcpy( h_weight, d_weight, weight_bytes, cudaMemcpyDeviceToHost )) ;
    cudaGetErrorString(cudaMemcpy( h_weight_b, d_weight_b, 1 * sizeof(float), cudaMemcpyDeviceToHost )) ;
    cudaGetErrorString(cudaMemcpy( h_dz, d_dz, predict_bytes, cudaMemcpyDeviceToHost )) ;

    for(int i = 0; i < weight_size; i++) printf("h_weight %f \n", h_weight[i]);
    // printf("h_weight_b %f \n", h_weight_b[0]);
    // for(int i = 0; i < 3500; i++) printf(" dz %f \n", h_dz[i]);

    // free GPU resource
    cudaGetErrorString(cudaFree( d_train_data )) ;
    cudaGetErrorString(cudaFree( d_weight )) ;
    cudaGetErrorString(cudaFree( d_predict )) ;
    cudaGetErrorString(cudaFree( d_softmax )) ;
    cudaGetErrorString(cudaFree( d_softmax_minus )) ;
    cudaGetErrorString(cudaFree( d_dz)) ;
    cudaGetErrorString(cudaDeviceReset()) ;

    //Test
    float accuracy = 0;
    Mult_CPU( h_train_data, h_weight, h_predict, h_weight_b, data_size, weight_size, 1) ;
    sigmoid_CPU( h_predict, h_softmax, data_size ) ;
    for (int i = 0; i < 3500; i++) {
        if((training_y[i][0]) == 1){
            if(h_softmax[i] > 0.5){
                accuracy++;
            }
        } 
        else{
            if(h_softmax[i] < 0.5){
                accuracy++;
            }
        }
    }
    printf("The testing accuracy is %f\n", accuracy/3500);
}
