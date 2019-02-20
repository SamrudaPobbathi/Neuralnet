#include <iostream> 
#include<omp.h>

using namespace std; 
  
#define N 4 
  
// This function multiplies  
// mat1[][] and mat2[][], and  
// stores the result in res[][] 
void multiply(int mat1[][N],  
              int mat2[][N],  
              int res[][N]) 
{ 
  int i, j, k, buffer=0; 

	

  #pragma omp parallel for private(j)
    for (i = 0; i < N; i++) 
    { 
        for (j = 0; j < N; j++) 
        { 
            res[i][j] = 0; 
		    	// buffer = 0;

            for (k = 0; k < N; k++) {

              res[i][j]+= mat1[i][k] *  
                             mat2[k][j]; 
            }
        // res[i][j]=buffer;

        } 
    } 
} 
  
// Driver Code 
int main() 
{ 
    int i, j; 
    double dtime;
    int res[N][N]; // To store result 
    int mat1[N][N] = {{1, 1, 1, 1}, 
                      {2, 2, 2, 2}, 
                      {3, 3, 3, 3}, 
                      {4, 4, 4, 4}}; 
  
    int mat2[N][N] = {{1, 1, 1, 1}, 
                      {2, 2, 2, 2}, 
                      {3, 3, 3, 3}, 
                      {4, 4, 4, 4}}; 
  	
	// clock_t start = clock();

	dtime = omp_get_wtime();                      

	// #pragma omp parallel
    multiply(mat1, mat2, res); 

    cout << "Result matrix is \n"; 
    for (i = 0; i < N; i++) 
    { 
        for (j = 0; j < N; j++) 
        cout << res[i][j] << " "; 
        cout << "\n"; 
    } 

	// clock_t stop = clock();
	dtime = omp_get_wtime() - dtime;
	// double elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    
	cout <<"\tTime elapsed:"<< dtime << 	endl;
  	
  
    return 0; 
} 