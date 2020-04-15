#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <algorithm>
using namespace std;

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <sys/time.h>
#include <unistd.h>

// time stamp function in seconds 
double getTimeStamp() {     
    struct timeval  tv ;     
    gettimeofday( &tv, NULL ) ;    
    return (double) tv.tv_usec/1000000 + tv.tv_sec ; 
} 

#define NUM_THREADS 1024

int features = 1024;
int sampels = 10000;
int classes = 10;

float ** training_x1; //3500 * 784
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
    training_x1 = (float**)malloc(sizeof(float*) * 10000);
    for(int i = 0; i < 10000; i++){
        training_x1[i] = (float*)malloc(sizeof(float) * 1024);
    }

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
        float temp=0;//reduce global mem access number
        result[threadId] = 0;
        for (int i = 0; i < N; i++)
        {
            //result[threadId] += a[row * N + i] * b[i * S + column];
            temp += a[row * N + i] * b[i * S + column];
        }
        result[threadId]=temp;
    }
}

__global__ void softmax_sum( float *predict, float *sum, const int label_size, const int data_size ){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data_size){
        float temp=0;
        for(int i = 0; i < label_size; i++){
            temp += exp(predict[tid * label_size + i]);
        }
        sum[tid]=temp;
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
            float temp = grad[tid * label_size + i];
            for(int j = 0; j < data_size; j++){
                // grad[tid * label_size + i] += train_data[j * weight_size + tid] * dz[j * label_size + i];
                temp += train_data[j * weight_size + tid] * dz[j * label_size + i];
            }
            grad[tid * label_size + i] = temp;
        }
    }
}

__global__ void weight_update(float *weight, float *grad, const int label_size, const int weight_size, const float learning_rate){
    int tid = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
              + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < weight_size){
        for(int i = 0; i < label_size; i++){
            grad[tid * label_size + i] /= 100;
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

int randint(int l,int u)
{
  int temp;
  srand((unsigned)time(NULL));
  temp = floor(l + (1.0*rand()/RAND_MAX)*(u - l + 1 ));
  return temp;
}


void random_shuffle(float *data, float *label){
    int len = 10000;
    for (int i = 0 ; i < len; i++) {
        int rand = randint(i, len - 1);
        // swap
        for(int j = 0; j < 1024; j++){
            //swap(data[i][j], arr[rand][j]);
            swap(data[i * 1024 + j], data[rand * 1024 + j]);
        }
        for(int k = 0; k < 10; k++){
            //swap(data[i][j], arr[rand][j]);
            swap(label[i * 10 + k], label[rand * 10 + k]);
        }
    }
}

void data_transpose(float *data1, float *data2){
    int batch_size = 200;
    int weight_size = 1024;
    int label_size = 10;
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < weight_size; j++){
            data2[j * batch_size+ i] = data1[i * weight_size + j];
        }
    }
}

void devide_data(float *data1, float *data2, float *label1, float *label2, int index){
    int batch_size = 200;
    int weight_size = 1024;
    int label_size = 10;
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < weight_size; j++){
            data1[i * weight_size + j] = data2[(index * batch_size + i) * weight_size + j];
        }
    }
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < label_size; j++){
            label1[i * label_size + j] = label2[(index * batch_size  + i) * label_size + j];
        }
    }
}


int main(){
    // Stream
    cudaDeviceProp prop;
	int deviceID;
	cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID);
    
	if (!prop.deviceOverlap)
	{
		printf("No device will handle overlaps. so no speed up from stream.\n");
		return 0;
    }
    
    //创建一个CUDA流
	cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStream_t stream2;
	cudaStreamCreate(&stream2);


    // malloc_host();
    malloc_host();
    readCSV("training_x.csv", training_x, 10000,1024);
    readCSV("training_y.csv", training_y, 1024, 1);
    readCSV("testing_x.csv", testing_x, 2000, 1024);
    readCSV("testing_y.csv", testing_y, 2000, 1);

    readCSV("training_x.csv", training_x1, 10000,1024);

    float learning_rate = 0.1;
    int iter = 1;
    int batch_size = 200;
    int epochs = 50;

    int data_size = 10000;
    int label_size = 10;
    int weight_size = 1024;

    int train_data_bytes = 10000 * 1024 * sizeof(float);
    int batch_data_bytes = 200 * 1024 * sizeof(float);
    int weight_bytes = 1024 * 10 * sizeof(float);
    int predict_bytes = 10000 * 10 * sizeof(float);
    int batch_predict_bytes = 200 * 10 * sizeof(float);
    

    float *h_train_data = (float *) malloc( train_data_bytes ) ;
    float *h_train_data_T = (float *) malloc( train_data_bytes ) ;
    float *h_batch_data = (float *) malloc( batch_data_bytes / 2) ;
    float *h_batch_data_T = (float *) malloc( batch_data_bytes ) ;
    float *h_label_onehot = (float *) malloc( predict_bytes ) ;
    float *h_batch_label = (float *) malloc( batch_predict_bytes ) ;

    float *h_weight = (float *) malloc( weight_bytes ) ;
    float *h_predict = (float *) malloc( batch_predict_bytes / 2 ) ;
    float *h_max = (float *) malloc( 100 * sizeof(float) ) ;
    float *h_sum = (float *) malloc( 100 * sizeof(float) ) ;
    float *h_softmax = (float *) malloc( batch_predict_bytes ) ;
    float *h_dz = (float *) malloc( batch_predict_bytes ) ;
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
    float *d_train_data,* d_weight, *d_predict1, *d_predict_sum1, *d_sum1, *d_max1, *d_softmax_value1, *d_dz1, *d_grad1;
    float *d_predict2, *d_predict_sum2, *d_sum2, *d_max2, *d_softmax_value2, *d_dz2, *d_grad2;
    float *d_batch_data1, *d_batch_label1, *d_batch_data2, *d_batch_label2;

    cudaGetErrorString(cudaMalloc( (void **) &d_train_data, train_data_bytes )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_weight, weight_bytes)) ;

    cudaGetErrorString(cudaMalloc( (void **) &d_batch_data1, batch_data_bytes / 2  )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_batch_label1, batch_predict_bytes / 2  )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_batch_data2, batch_data_bytes / 2   )) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_batch_label2, batch_predict_bytes / 2  )) ;
    
    cudaGetErrorString(cudaMalloc( (void **) &d_predict1, batch_predict_bytes / 2)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_predict2, batch_predict_bytes / 2)) ;

    cudaGetErrorString(cudaMalloc( (void **) &d_sum1, 100 * sizeof(float))) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_softmax_value1, batch_predict_bytes / 2)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_dz1, batch_predict_bytes / 2)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_grad1, weight_bytes)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_max1, 100 * sizeof(float))) ;

    cudaGetErrorString(cudaMalloc( (void **) &d_sum2, 100 * sizeof(float))) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_softmax_value2, batch_predict_bytes / 2)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_dz2, batch_predict_bytes / 2)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_grad2, weight_bytes)) ;
    cudaGetErrorString(cudaMalloc( (void **) &d_max2, 100 * sizeof(float))) ;

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
    double timeStamp1 = getTimeStamp() ;
    for(int train  = 0; train < 1; train++){
        //////////////////////Random shuffle data/////////////////////////////
        random_shuffle(h_train_data, h_label_onehot);
    
        for(int epoch = 0; epoch < epochs; epoch++){
            //////////////////////   Transfer data   ////////////////////////////
            devide_data(h_batch_data, h_train_data, h_batch_label, h_label_onehot, epoch);
            cudaGetErrorString(cudaMemcpyAsync( d_weight, h_weight, weight_bytes, cudaMemcpyHostToDevice)) ;
            cudaGetErrorString(cudaDeviceSynchronize());
            cudaGetErrorString(cudaMemcpyAsync( d_batch_data1, h_batch_data, batch_data_bytes / 2, cudaMemcpyHostToDevice, stream1 )) ;
            cudaGetErrorString(cudaMemcpyAsync( d_batch_label1, h_batch_label, batch_predict_bytes / 2, cudaMemcpyHostToDevice, stream1 )) ;
            cudaGetErrorString(cudaMemcpyAsync( d_batch_data2 , h_batch_data + 100 * 1024, batch_data_bytes / 2, cudaMemcpyHostToDevice, stream2 )) ;
            cudaGetErrorString(cudaMemcpyAsync( d_batch_label2, h_batch_label + 100 * 10, batch_predict_bytes / 2, cudaMemcpyHostToDevice, stream2 )) ;
            

            //////////////////////     Computation    ///////////////////////////
            //Initialize
            initialize<<<gridDim, blockDim, 0, stream1>>>(d_sum1, d_predict1, batch_size / 2 , label_size);
            initialize_dz<<<gridDim, blockDim, 0, stream1>>>(d_dz1, label_size, batch_size / 2);
            initialize_grad<<<gridDim, blockDim, 0, stream1>>>(d_grad1, label_size, weight_size);

            initialize<<<gridDim, blockDim, 0, stream2>>>(d_sum2, d_predict2, batch_size / 2, label_size);
            initialize_dz<<<gridDim, blockDim, 0, stream2>>>(d_dz2, label_size, batch_size / 2);
            initialize_grad<<<gridDim, blockDim, 0, stream2>>>(d_grad2, label_size, weight_size);


            // //DOT
            Mult_GPU<<<gridDim, blockDim, 0, stream1>>>( d_batch_data1, d_weight, d_predict1, batch_size / 2, weight_size, label_size) ;
            Mult_GPU<<<gridDim, blockDim, 0, stream2>>>( d_batch_data2, d_weight, d_predict2, batch_size / 2, weight_size, label_size) ;

            max<<<gridDim, blockDim, 0, stream1>>>( d_predict1, d_max1, label_size, batch_size / 2 );
            max<<<gridDim, blockDim, 0, stream2>>>( d_predict2, d_max2, label_size, batch_size / 2 );

            normalize<<<gridDim, blockDim, 0, stream1>>>(d_predict1, d_max1, label_size, batch_size / 2 );
            normalize<<<gridDim, blockDim, 0, stream2>>>(d_predict2, d_max2, label_size, batch_size / 2 );

            // // Softmax
            softmax_sum<<<gridDim, blockDim, 0, stream1>>>( d_predict1, d_sum1, label_size, batch_size / 2  );
            softmax<<<gridDim, blockDim, 0, stream1>>>( d_softmax_value1, d_predict1, d_sum1, label_size, batch_size / 2  );
            softmax_sum<<<gridDim, blockDim, 0, stream2>>>( d_predict2, d_sum2, label_size, batch_size / 2  );
            softmax<<<gridDim, blockDim, 0, stream2>>>( d_softmax_value2, d_predict2, d_sum2, label_size, batch_size / 2  );


            // // Weight Update
            dz<<<gridDim, blockDim, 0, stream1>>>(d_softmax_value1, d_batch_label1, d_dz1, label_size, batch_size / 2 );
            grad<<<gridDim, blockDim, 0, stream1>>>(d_batch_data1, d_dz1, d_grad1, label_size, batch_size / 2 , weight_size);
            dz<<<gridDim, blockDim, 0, stream2>>>(d_softmax_value2, d_batch_label2, d_dz2, label_size, batch_size / 2 );
            grad<<<gridDim, blockDim, 0, stream2>>>(d_batch_data2, d_dz2, d_grad2, label_size, batch_size / 2 , weight_size);

            //等待Stream流执行完成
            // cudaStreamSynchronize(stream1);
	        // cudaStreamSynchronize(stream2);
            weight_update<<<gridDim, blockDim, 0, stream1>>>(d_weight, d_grad1, label_size, weight_size, learning_rate);
            weight_update<<<gridDim, blockDim, 0, stream2>>>(d_weight, d_grad2, label_size, weight_size, learning_rate);
            cudaGetErrorString(cudaDeviceSynchronize());
       }
   }

    cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);
    // double timeStamp2 = getTimeStamp() ;

    cudaGetErrorString(cudaMemcpyAsync( h_weight, d_weight, weight_bytes, cudaMemcpyDeviceToHost)) ;
    for(int i = 0; i < weight_size; i++){
        for(int j = 0; j < label_size; j++){
            printf("h_weight: %f\n", h_weight[i * label_size + j]);
        }
    }


    // cudaStreamDestroy(stream1);
    // cudaStreamDestroy(stream2);

    // ///////////////////////////// Test Case /////////////////////////////////////
    // cudaGetErrorString(cudaMemcpyAsync( h_sum, d_sum1, 100 * sizeof(float), cudaMemcpyDeviceToHost)) ;
    // for(int i = 0; i < 100; i++){
    //     printf("sum : %f\n", h_sum[i]);
    // }

    // cudaGetErrorString(cudaMemcpyAsync( h_predict, d_softmax_value2, batch_predict_bytes / 2, cudaMemcpyDeviceToHost)) ;
    // for(int i = 0; i < 100; i++){
    //     for(int j = 0; j < label_size; j++){
    //         printf(" h_predict: %f\n",  h_predict[i * label_size + j]);
    //     }
    // }

    // cudaGetErrorString(cudaMemcpyAsync( h_batch_data, d_batch_data2, batch_data_bytes / 2, cudaMemcpyDeviceToHost)) ;
    // for(int i = 0; i < 100; i++){
    //     for(int j = 0; j < weight_size; j++){
    //         printf(" h_train_data: %f\n",  h_batch_data[i * weight_size + j]);
    //     }
    // }


    // float *h_label = (float *) malloc( batch_predict_bytes / 2 ) ;
    // cudaGetErrorString(cudaMemcpyAsync( h_label, d_batch_label1, batch_predict_bytes / 2, cudaMemcpyDeviceToHost)) ;

    // for(int i = 0; i < batch_size / 2; i++){
    //     for(int j = 0; j < label_size; j++){
    //         printf("label %f\n", h_label[i * label_size + j]);
    //     }
    // }
    //////////////////////////////// Data Transfer Success ////////////////////////////////


    // cudaGetErrorString(cudaMemcpyAsync( h_weight, d_weight, weight_bytes, cudaMemcpyDeviceToHost)) ;
    // for(int i = 0; i < weight_size; i++){
    //     for(int j = 0; j < label_size; j++){
    //         printf("h_weight: %f\n", h_weight[i * label_size + j]);
    //     }
    // }

    // printf("%.6f\n", timeStamp2-timeStamp1);

    // // Test case
    // for(int i = 0; i < data_size; i++){
    //     for(int j = 0; j < weight_size; j++){
    //         h_train_data[i * weight_size + j] = training_x1[i][j];
    //         //printf(" h_train_data: %f\n",  h_train_data[i * label_size + j]);
    //     }
    // }

    // float *h_test_predict = (float *) malloc( predict_bytes ) ;

    // float *d_test_predict, *d_test_max;
    // cudaGetErrorString(cudaMalloc( (void **) &d_test_predict, predict_bytes)) ;
    // cudaGetErrorString(cudaMalloc( (void **) &d_test_max, 10000 * sizeof(float))) ;

    // cudaGetErrorString(cudaMemcpy( d_train_data, h_train_data, train_data_bytes, cudaMemcpyHostToDevice )) ;
    // cudaGetErrorString(cudaMemcpy( d_weight, h_weight, weight_bytes, cudaMemcpyHostToDevice )) ;

    // Mult_GPU<<<gridDim, blockDim>>>( d_train_data, d_weight, d_test_predict, data_size, weight_size, label_size) ;
    // cudaGetErrorString(cudaDeviceSynchronize());
    // // max<<<gridDim, blockDim>>>( d_test_predict, d_test_max, label_size, data_size );
    // // cudaGetErrorString(cudaDeviceSynchronize());	
    // // normalize<<<gridDim, blockDim>>>(d_test_predict, d_test_max, label_size, data_size);
    // // cudaGetErrorString(cudaDeviceSynchronize());

    // cudaGetErrorString(cudaMemcpy( h_test_predict, d_test_predict, predict_bytes, cudaMemcpyDeviceToHost )) ;

    // for(int i = 0; i < 10000; i++){
    //     for(int j = 0; j < 10; j++){
    //         if(i == 0 )printf("h_predict: %f\n", h_predict[i * label_size + j]);
    //     }
    // }



    ///////////////////////// Error ///////////////////////////////
    // float total_error = 0;
    // for(int i = 0; i < batch_size; i++){
    //     for(int j = 0; j < label_size; j++){
    //         total_error -= label_onehot[i][j] * log(h_softmax[i * label_size + j]) ;
    //     }
    // }
    // printf("error: %f\n", total_error );

}