#ifndef DATA_LOADER_C
#define DATA_LOADER_C


struct  data_struct
{
  float **data_images ;
  int *data_labels;
};

struct data_struct data_loader(int,char[]);

#endif 