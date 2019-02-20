**Illinois Institute of Technology**  
**CS554 Data-Intensive Computing (Fall 2018)**  
**uDNN Deep Learning Project**  
Project mentor: Alexandru Iulian Orhean (aorhean@hawk.iit.edu)  
Team name: Micro Deep Neural Networks with Variable Prescision Computations  
Team members:  
*  Samruda Pobbathi (spobbathi@hawk.iit.edu)
*  Manuel Alaman Escolano (malamanescolano@hawk.ii.edu)

The determination of the influence of different resolutions of data precision in the performance of neural networks is the main goal of this project. The influence of CPU and GPU for the computing of the such variable precision operations is another branch that is going to be explored. As deep neural networks consists of huge amount of computation that can be parallelized, we can improve the performance by exploiting this possibility on GPUs. Once accomplished that, other implementations of neural networks, principally deeper networks and developed in other languages, will be performed in order to compare the effects of the precision variations.


##USE THE FOLLOWING COMMANDS TO RUN THE PROJECT INSIDE THE /src FOLDER
gcc -c -o data_loader.o data_loader.c 
g++ -O3 -Wall -c -o train.o train.cpp -fopenmp -lblas
 g++ -o train data_loader.o train.o -fopenmp -lblas

##To run with the mnist data: 
./train m

##To run with the fmnist data:
./train f

##TO RUN WITH SINGLE THREAD, RUN THE BELOW COMMAND AND THEN COMPILE WITHOUT fopenmp flag for the second g++ command:
export OMP_NUM_THREADS=1
