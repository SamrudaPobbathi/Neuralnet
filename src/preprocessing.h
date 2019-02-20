#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void preprocess(float test_image[][784], float train_image[][784] ){
    float sum_total = 0;
    float sum_image = 0;
    float avg,variance,std_dev;
// //Test
//     for(int i = 0; i < 10000; i++){
    
//         for (int j = 0; j < 784; j++){
//             //Correct syntax
//             sum_image = sum_image + test_image[i][j];
//         }
//         sum_total = sum_total+sum_image;
//     }
//     avg = sum_total/(10000*784);
//     // printf("Average:%f\n",avg);

//     int sum1 = 0;
//     for(int i = 0; i < 10000; i++){
//         for (int j = 0; j < 784; j++){
//             sum1 = sum1 + pow((test_image[i][j]-avg),2);
//         }
//         sum_total = sum_total+sum_image;
//     }
//     variance = sum1/(float)(10000*28*28);
//     std_dev = sqrt(variance);
//     // printf("Standard deviation:%f\n",std_dev);
// //Substract the average
// //Divide by the std deviation

//     for(int i = 0; i < 10000; i++){
//         for (int j = 0; j < 784; j++){
//             //Correct syntax
//             test_image[i][j]-=avg;
//             test_image[i][j]/=std_dev;
//         }
//     }

}
