#include<iostream>
#include <string>
#include "feedforward.h"
#include <time.h>
#include<omp.h>
#include <chrono>

extern "C" {
  #include "data_loader.h" //a C header, so wrap it in extern "C" 
}

using namespace std;
#define IMG_SIZE 784
#define learning_rate 0.001
#define num_images_to_train 4000
#define eval_every 200


template <class T>
T find_max(T *arr,T length){
	T max = 0,label;
	for(int i=0; i<length; i++){
		if(arr[i] > max){
			max = arr[i];
			label = i;
		}

	}
	return label;
}

template <class T>
// T eval_nn(FeedForward<float>  nn,T **imgs,int *labels,int num_images){
T eval_nn(FeedForward<double>  nn,T **imgs,int *labels,int num_images){
// T eval_nn(FeedForward<long double>  nn,T **imgs,int *labels,int num_images){

	int correct_no = 0;
	T predicted,actual,*buffer_array;
	buffer_array = new T[IMG_SIZE];

	//Change this to 784 now it is doing only for one image
	for(int i=0; i<num_images; i++){
		//Now the output of last layer is 10 this length = 10
		for(int j=0; j< IMG_SIZE; j++){
			buffer_array[j] = imgs[i][j];
			// cout << buffer_array[j] <<endl;
		}
		// predicted = find_max<float>(nn.forward(buffer_array),10);
		predicted = find_max<double>(nn.forward(buffer_array),10);
		// predicted = find_max<long double>(nn.forward(buffer_array),10);


		actual = labels[i];

		if((actual - predicted) < 0.5){
			correct_no+= 1; 
		}

	}
	return (T)(correct_no) / (T)(num_images);
}

template <class T>
// int train_nn(FeedForward<float> nn,T **data,T **test_data,int *labels,int *test_labels){
int train_nn(FeedForward<double> nn,T **data,T **test_data,int *labels,int *test_labels){
// int train_nn(FeedForward<long double> nn,T **data,T **test_data,int *labels,int *test_labels){


	unsigned int i,j;
	T accuracy,test_accuracy;
	T *nn_output, *expected_output, *buffer_array, *error;
    // double dtime; 
	buffer_array = new T[IMG_SIZE];
	

	// clock_t start = clock();
	// dtime = omp_get_wtime();   
    auto t1 = std::chrono::high_resolution_clock::now();
                   

	//Change this to 784 now it is doing only for one image
	for(i=0; i< num_images_to_train; i++){
		//Extracting and sending the data row-wise
		for(j=0; j< IMG_SIZE; j++){
			buffer_array[j] = data[i][j];
			// cout << buffer_array[j] <<endl;
		}
		nn_output = nn.forward(buffer_array);

		error = new T[sizeof(nn_output)];
		expected_output = new T[sizeof(nn_output)];

		//Initializing the expected output to have zeros for the size of the output array so that the index that has label is made 1
		for(j=0; j<= sizeof(nn_output); j++){
			expected_output[j] = 0;
		}
		expected_output[labels[i]] =  1;
		for(j=0; j<= sizeof(nn_output); j++){
			error[j] =  abs(expected_output[j] - nn_output[j]);
		}
		nn.backward(buffer_array, error);
		nn.update_parameters(learning_rate);

		//Change 1 to number of images
		if(i % eval_every == 0){
			// accuracy = eval_nn<float>(nn, data, labels, num_images_to_train);
			// test_accuracy = eval_nn<float>(nn, test_data, test_labels, num_images_to_train);

			accuracy = eval_nn<double>(nn, data, labels, num_images_to_train);
			test_accuracy = eval_nn<double>(nn, test_data, test_labels, num_images_to_train);

			// accuracy = eval_nn<long double>(nn, data, labels, num_images_to_train);
			// test_accuracy = eval_nn<long double>(nn, test_data, test_labels, num_images_to_train);

			// clock_t stop = clock();
			// dtime = omp_get_wtime() - dtime;
			auto t2 = std::chrono::high_resolution_clock::now();

			double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

			// double elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
			// cout << "training accuracy :" << accuracy << "\t"<< "test accuracy :" << test_accuracy << "\t"<<"time elapsedin ms:"<< elapsed << 	endl;
			// cout <<"Train accuracy:"<<  accuracy << "\tTrain accuracy:"<<  test_accuracy << "\tTime elapsed in ms: "<< elapsed << 	endl;
			// cout << "\t"<<accuracy <<"\t"<<  test_accuracy <<"\t"<< elapsed << 	endl;
			cout << "\t"<<accuracy <<"\t"<<  test_accuracy <<"\t"<< elapsed << 	endl;


		}
	}
	
	
	return 0;	
}

int main(int argc, char** argv){

	struct data_struct train_data;
	struct data_struct test_data;
	test_data = data_loader(1,argv[1]);

  	layers_info_data layers_info[3];
	layers_info[0].layer_size=300;
	layers_info[0].transfer_function = "logistic";
	layers_info[0].vectorize = "TRUE";


	// cout<<argv[0]<<argv[1]<<endl;
	// layers_info[1].layer_size=100;
	// layers_info[1].transfer_function = "hyperbolic_tangent";

	layers_info[1].layer_size=10;
	layers_info[1].transfer_function = "identity";
	layers_info[1].vectorize = "TRUE";

  	// FeedForward<float> nn(IMG_SIZE,layers_info);
  	FeedForward<double> nn(IMG_SIZE,layers_info);
  	// FeedForward<long double> nn(IMG_SIZE,layers_info);


	train_data = data_loader(0,argv[1]);
	// train_nn<float>(nn, train_data.data_images, test_data.data_images,train_data.data_labels, test_data.data_labels);

	train_nn<double>(nn, (double**)(train_data.data_images), (double**)(test_data.data_images),train_data.data_labels, test_data.data_labels);

	// train_nn<long double>(nn, (long double**)(train_data.data_images), (long double**)(test_data.data_images),train_data.data_labels, test_data.data_labels);



  return 0;
}

