#include<iostream>
#include <math.h>  
#include <cmath>  
#include<omp.h>


template <typename T> 
T* identity(T *x, int size, bool derivative=false){
	T *res;
    // int n = (int)(sizeof(x)/sizeof(*x));
    int n,i;
    n = size;
    res = new T[n];

    // omp_set_num_threads(2);
	if(derivative){
        #pragma omp parallel for shared(res,i,n) 
        // #pragma omp parallel
		for(i=0; i<n ; i++){
			res[i] = 1;
		}
	}
	else{
        #pragma omp parallel for shared(res,i,x,n) 
		for(i=0; i<n; i++){
			res[i] = x[i];
		}
	}
	return res;
}

template <typename T> 
T* logistic(T *x,int size,bool derivative = false){
    T *res;
    // int n = (int)(sizeof(x)/sizeof(*x));
    
    int n,i;
    n = size;

    res = new T[n];

    // omp_set_num_threads(8);

    if (derivative){
        #pragma omp parallel for shared(res,i,x) 

		for(i=0; i<n; i++){
        	res[i] =  x[i] * (1 - x[i]);
            // std::cout << res[i];

        }
    }
    else{
        #pragma omp parallel for shared(res,i,x) 
		for(i=0; i<n; i++){
        	res[i] =  1 / (1 + exp(-x[i]));
            // std::cout <<  -x[i];
    	}
    }

    return res;
}

template <typename T> 
T* hyperbolic_tangent(T *x, int size, bool derivative = false){
    T *res;
    // int n = (int)(sizeof(x)/sizeof(*x));
    int n,i;
    n = size;
    res = new T[n];

    if(derivative){
        #pragma omp parallel for shared(res,i,x) 

		for( i=0; i<n; i++){

        	res[i] =  1 - pow(x[i],2);
    	}
    }
    else{
        #pragma omp parallel for shared(res,i,x) 

    	// res = (np.e ** x - np.e ** (-x)) / (np.e ** x + np.e ** (-x))
		for(i=0; i<n; i++){


        	res[i] = (exp(x[i]) - exp(-x[i])) / (exp(x[i]) + exp(-x[i]));
    	}
    }

    return res;

}
