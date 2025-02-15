#pragma once
#include "/usr/local/cuda/include/cuda_runtime.h"

////////////////////// GPU P2P DATA ///////////////////

//device arrays
extern double* d_particle_data;
extern int* d_leaf_data;
extern int* d_interaction_data;
extern double* d_result_data;

//control memory
extern int max_particle_data;
extern int max_leaf_data;
extern int max_interaction_data;
extern int max_result_data;
///////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" 
{
#endif

void initGPU(int verbosity_gpu);

//interactions with neighbor boxes
void getGPUMemoryState(int verbosity_gpu);
int  allocMemGPU(int nleafs, int maxPartsInLeaf, int maxNeighbors, int nTasks, int verbosity_gpu);
int  copyMemGPU(double* h_pos_data, int* h_leaf_data, int* h_interactions, int numtasks, int verbosity_gpu);
void readResultsGPU(double* h_acc_data, int nTasks, int maxPartsInLeaf, int verbosity_gpu);
int  LaunchKernelP2PIndexing(
            int ntasks, int posChunk, int leafChunk, int resultChunk, 
            double SoftenScale, double MASSPART, int verbosity_gpu);
__global__ void ComputeP2PIndexing(
            double* pos_data, int* leaf_data, int* interaction_data, double* result_data, 
            int ntasks, int posChunk, int leafChunk, int resultChunk,
            double SoftenScale, double MASSPART, int* kernelError,
            int max_particle_data, int max_leaf_data, int max_interaction_data, int max_result_data, int threadGroup);

#ifdef __cplusplus
}
#endif