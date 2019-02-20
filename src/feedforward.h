#include<iostream>
#include<string>
#include "layer.h"
#include <vector> 
#include<omp.h>

using namespace std;

#define NUM_LAYERS 2
//TO PASS THE INFO OF EACH LAYER AT THE TIME OF CONSTRUCTOR INIT (tuple)
struct layers_info_data{
	int layer_size; 
	bool vectorize;
	string transfer_function;  //INCREASE THE SIZE IF THE NAME OF THE TRANSFER FUNCTION IS LONG
};

template <typename T> 
class FeedForward{
	public:
		//Layers two but change it to generic number for more layers
		// Layer<float> *layers;
		Layer<double> *layers;


	public:FeedForward(){

	}

	public:FeedForward(int input_size,layers_info_data* layers_info,int total_nn_layers=NUM_LAYERS){
		int last_size;
		last_size = input_size;
		// this->layers = new Layer<float>[NUM_LAYERS];
		this->layers = new Layer<double>[NUM_LAYERS];

		
		#pragma omp parallel for 
		for(int i=0;i<total_nn_layers;i++){
			this->layers[i].Layer_initiator(last_size,layers_info[i].layer_size, layers_info[i].transfer_function,layers_info[i].vectorize);
			last_size = layers_info[i].layer_size;
		}
		
	}
	//DO it once for all the layers
	T* forward(T *inputs){
		int i;
		T *last_input = inputs;		
		for(i=0;i<NUM_LAYERS;i++){
			last_input = this->layers[i].forward(last_input);
		}
		return last_input;
	}

	void backward(T *inputs,T *output_error){
		int i;
		// Layer<float> crt_layer,prev_layer;
		Layer<double> crt_layer,prev_layer;

		T *crt_error;

		crt_error = output_error;

		for(i = NUM_LAYERS-1; i>0 ; i=i-1){
			crt_layer = this->layers[i];
			prev_layer = this->layers[i-1];
			crt_error = crt_layer.backward(prev_layer.outputs, crt_error); 
		
		}
		this->layers[0].backward(inputs, crt_error);
	}
	void update_parameters(T learning_rate){
		int i,j,k;
		T **g_weight;
		int rows; 
		int cols; 

		for(i=0; i< NUM_LAYERS; i++){
			rows = this->layers[i].num_output;
			cols = this->layers[i].num_input;
			g_weight = new T*[rows];
			for(j=0; j< rows  ; j++)
				g_weight[j] = new T[cols];

			for(j=0; j< rows  ; j++){
				for (k = 0; k < cols ; k++)
				{
					/* code */
					g_weight[j][k] = learning_rate * this->layers[i].g_weights[j][k];
					this->layers[i].weights[j][k] = this->layers[i].weights[j][k] - g_weight[j][k];
				}
				
			}
		}
	}
};
	
