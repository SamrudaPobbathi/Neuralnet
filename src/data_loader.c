#include "mnist.h"
// #include "preprocessing.h"

#include <stdio.h>
#include <math.h>

#include "data_loader.h"

struct data_struct data_loader(int type,char data_type[]){
  int Rows,Cols = 784;
  if(type==0){ Rows = 60000;}
  else if(type==1){ Rows = 10000;}
  struct data_struct data;
  data.data_images = (float **)malloc(Rows * sizeof(float *));
  data.data_labels = (int *)malloc(Rows * sizeof(int));
  // int *data_labels;

  for (int row = 0; row < Rows; row++) {
      data.data_images[row] = (float *)malloc(784 * sizeof(float));
  }
  load_mnist(data_type);
  // preprocess(test_image,train_image);

  if(type == 0){
    for(int i=0; i< Rows;i++){
      for(int j=0;j < Cols;j++){
         data.data_images[i][j] = train_image[i][j]; 
      }
      data.data_labels[i] = train_label[i];
      // printf("%d\n", data.data_labels[i]);
    }
  }
  else if(type==1){
    for(int i=0; i< Rows;i++){
      for(int j=0;j < Cols;j++){
         data.data_images[i][j] = test_image[i][j]; 
      }
      data.data_labels[i] = test_label[i];

    }
  }

  return data;
}


