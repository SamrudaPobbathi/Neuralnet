#include<iostream>
#include<string>
#include<random>
#include<math.h>
#include"transfer_functions.h"
#include <cmath>
#include<omp.h>
#include </usr/local/Cellar/openblas/0.3.3/include/cblas.h>
// #include <gsl/gsl_blas.h>

using namespace std;


template <typename T> 
 //Abstract class for transfer functions
class Layer 
{ 
    // Layer constructor, that receives as a parameter the number of inputs, the
      // number of outputs and a sigmoid transfer function

    public: 
        int num_input;
        int num_output;
        T **weights;
        T *biases;
        T *outputs;
        T *g_biases;
        T **g_weights;
        T *g_outputs;
        bool vectorize;
        // char transfer_function[];
        string transfer_function;
        

    // Member Functions() 
    public:Layer(){

    }
     // return a uniformly distributed random number
    double RandomGenerator()
    {
      return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
    }
    // return a normally distributed random number
    double normalRandom()
    {
      double y1=RandomGenerator();
      double y2=RandomGenerator();
      return cos(2*3.14*y2)*sqrt(-2.*log(y1));
    }

    void Layer_initiator(int num_input, int num_output, string transfer_function,bool vectorize) 
    { 
      T number;
      int i,j;
      //Initializning all the vectors:
      this->biases = new T[num_output];
      this->outputs = new T[num_output];
      this->g_biases = new T[num_input];
      this->g_weights = new T*[num_output];
      this->g_outputs = new T[num_output];
      this->weights = new T*[num_output];
      this->vectorize = vectorize;

      // #pragma omp parallel private(i) 
      // #pragma omp parallel for 
      for(int i=0;i<num_output;i++){
        this->weights[i] = new T[num_input];
        this->g_weights[i] = new T[num_input];
      }


      this->num_input = num_input;
      this->num_output = num_output;
      // cout << this->num_output << "num of outputs" << this->num_input <<"num of inputs "<<endl;

        // strcpy(this->transfer_function,transfer_function);
        this->transfer_function = transfer_function;
        T mean = 0.0;
        T standard_deviation = sqrt( 2.0/ T(this->num_output + this->num_input));

      std::default_random_engine generator;
      std::normal_distribution<double> distribution(mean,standard_deviation);

      
      for (i=0;i<this->num_output;i++){
          // #pragma omp parallel for 
        for(j=0;j<this->num_input;j++){
          number = distribution(generator);
          this->weights[i][j] = number;
          this->g_weights[i][j] = 0;
        }
      }

      //Updating Biases and initializing global outputs, biases and outputs of this layer.
      // #pragma omp parallel private(number,i) 
      #pragma omp parallel for  
      for (i=0;i<this->num_output;i++){
          number = distribution(generator);
          // this->weights[i][0] = number;
          this->biases[i] = number;
          this->outputs[i] = 0;
          this->g_biases[i] = 0;
          this->g_outputs[i] = 0;
      }

  } 

  T **dotProduct(T *vect_B,T **vect_A,int row1,int col1,int row2,int col2) 
  { 
     int i,j,k,q,t;
     T **product,buffer; 

      if(row1 == this->num_output && col1 == 1 && row2 == 1 && col2 == this->num_input){
        product = new T*[this->num_output];
        for (i = 0; i < this->num_output; i++) {
          product[i] = new T[this->num_input]; 
          
        }
        
        if(this->vectorize){
          // Without using BLAS
          // Loop for calculate dot product 
            #pragma omp parallel for shared(product,vect_A, vect_B)

            for (i = 0; i < this->num_output; i++) {
              for (j = 0 ; j < this->num_input; j++ ){
                  product[i][j] = vect_B[i] * vect_A[0][j] ;       
              }
              
            }
        }
        else{
        //WITH using BALS
          double *B;
            B = new double[this->num_input];

            double *output;
            output = new double[this->num_output*this->num_input];
       

          #pragma omp parallel for private(t)
            for (q = 0; q < 1; q++)
            {
              for (t = 0; t < this->num_input; t++)
              {
                B[q * 1 + t] =   (double)vect_A[q][t];
              }
            }

          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            this->num_output,this->num_input,1,
            1.0, (double*)vect_B, 1, B ,this->num_input, 0.0, output, this->num_input);


          #pragma omp parallel for private(j)
          for (i = 0; i < this->num_output; i++) {
            for(j=0; j< this->num_input; j++) {          
              product[i][j] = (output[this->num_input * i + j]);
            }
          }
        }

      // delete[] B;
      // delete[] output;

      return product; 

      }
      else if(row1 == this->num_input && col1 == this->num_output && row2 == this->num_output && col2 == 1){
        product = new T*[this->num_input];

        for (i = 0; i < this->num_input; i++) {
          product[i] = new T[1]; 
        }

        for(i=0; i< this->num_input; i++){
          product[i][0] = 0;
        }

        

          #pragma omp parallel for shared(buffer,vect_A, vect_B)

          for(i=0; i<this->num_input; i++){
            for(j=0; j< 1; j++){
              buffer = 0;
              for(k=0; k<this->num_output; k++){
                    buffer+= (vect_A[i][k] * vect_B[k]);
                  }
              product[i][j] = buffer;
            }
          }

           //WITH using BALS
        // double *B;
        // B = new double[this->num_input*this->num_output];

        // double *output;
        // output = new double[this->num_input];
   

        // #pragma omp parallel for private(t)
        //   for (q = 0; q < this->num_input; q++)
        //   {
        //     for (t = 0; t < this->num_output; t++)
        //     {
        //       B[q * 1 + t] =   (double)vect_A[q][t];
        //     }
        //   }

        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        //   this->num_input,1,this->num_output,
        //   1.0, B, this->num_output, (double*)vect_B ,1, 0.0, output, 1);


        // #pragma omp parallel for private(j)
        // for (i = 0; i < this->num_input; i++) {
        //   for(j=0; j< 1; j++) {          
        //     product[i][j] = output[i];
        //   }
        // }

        // delete[] B;
        // delete[] output;

       return product; 

      }
      else if(row1 == this->num_output && col1 == this->num_input && row2 == this->num_input && col2 == 1){
        product = new T*[this->num_output];

        for (i = 0; i < this->num_output; i++) {
          product[i] = new T[1]; 
        }

        for(i=0; i< this->num_output; i++){
          product[i][0] = 0;
        }

        { 
          #pragma omp parallel for shared(buffer,vect_A, vect_B)
          for(i=0; i< this->num_output; i++){
            for(j=0; j< 1; j++){
              buffer = 0;
              for(k=0; k<this->num_input; k++){
                    buffer+= (vect_A[i][k] * vect_B[k]);
                  }
                  product[i][j] = buffer;
            }
          }
        }
       return product; 

      }
      return 0 ;
  }

  T *dotProduct(T *vect_A, T *vect_B,int length) 
  { 
     // int n = sizeof vect_B +1;
      T *product; 
      int i;
      product = new T[length];

      // Loop for calculate dot product 
      // #pragma omp parallel for shared(product,vect_A,vect_B)
      for (i = 0; i < length; i++) {
        product[i] = vect_A[i] * vect_B[i]; 
      }

      return product; 
  }

  void transposeMatrix(T **matrixA,T **transpose,int rows,int cols){
    int i,j;

    #pragma omp parallel for shared(transpose,matrixA)
    for(i=0;i<rows;i++){
      for(j=0;j<cols;j++){
          transpose[i][j] = matrixA[j][i];
      }
    }
  }


  void transposecolVector(T *vect_A,T **transposeMatrix){

    int i;
      // #pragma omp parallel for 
      for(i=0; i < this->num_input; i++){
          transposeMatrix[0][i] = vect_A[i]; 
      }

  }

  T *transposeMattoColVect(T **transposeMatrix,int length){
      T *vect_A;
      int i;
      vect_A =  new T[length];

      // #pragma omp parallel for shared
      for(i=0; i<length; i++){
         vect_A[i] = transposeMatrix[i][0]; 
      }

    return vect_A;
  }

  T* forward(T *inputs){
      this->outputs = transposeMattoColVect(dotProduct( inputs, this->weights, this->num_output,this->num_input,this->num_input,1), this->num_output);
    
      if(this->transfer_function.compare("identity")){
        this->outputs = identity(this->outputs,this->num_output);
      }
      else if(this->transfer_function.compare("logistic")){

        this->outputs = logistic(this->outputs,this->num_output);  
      }
      else if(this->transfer_function.compare("hyperbolic_tangent")){

        this->outputs = hyperbolic_tangent(this->outputs,this->num_output);  
      }


      return this->outputs;
    }

    T* backward(T *inputs,T *errors){

        //transpose matrix to store the transpose of input matrix while calculating weights
        T **transposedMatrix;
        T *result;
        transposedMatrix = new T*[1];
        //current input size equals previous layer's output
        transposedMatrix[0] = new T[this->num_input];


      if(this->transfer_function.compare("identity")){
          // this->g_outputs = identity<float>(this->outputs,this->num_output,true);
          this->g_outputs = identity<double>(this->outputs,this->num_output,true);
          // this->g_outputs = identity<long double>(this->outputs,this->num_output,true);


        }
      else if(this->transfer_function.compare("logistic")){
          // this->g_outputs = logistic<float>(this->outputs,this->num_output,true);  
          this->g_outputs = logistic<double>(this->outputs,this->num_output,true);  
          // this->g_outputs = logistic<long double>(this->outputs,this->num_output,true);  


        }
      else if(this->transfer_function.compare("hyperbolic_tangent")){
          // this->g_outputs = hyperbolic_tangent<float>(this->outputs,this->num_output,true);  
          this->g_outputs = hyperbolic_tangent<double>(this->outputs,this->num_output,true);  
          // this->g_outputs = hyperbolic_tangent<long double>(this->outputs,this->num_output,true);  


        }
        

        this->g_biases = dotProduct(errors, this->g_outputs, this->num_output);


        //transposing input col vector to find the weight matrix
        transposecolVector(inputs,transposedMatrix);
       
        this->g_weights = dotProduct(dotProduct(errors, this->g_outputs,this->num_output),transposedMatrix,this->num_output,1,1,this->num_input);
        
        delete[] transposedMatrix[0];
        delete[] transposedMatrix;

        //To store weight matrix's transpose: weight mat-- num_outputs*num_inputs
        transposedMatrix = new T*[this->num_input];
        for(int i=0; i<this->num_input; i++){
                  transposedMatrix[i] = new T[this->num_output];
        }
        transposeMatrix(this->weights,transposedMatrix,this->num_input,this->num_output);

 
        result = new T[this->num_input];
        result = transposeMattoColVect(dotProduct(this->g_biases,transposedMatrix,this->num_input,this->num_output,this->num_output,1),this->num_input);

        // for(int i=0;i<this->num_input;i++)
        //   cout<<result[i];
        return result;

      }

}; 
