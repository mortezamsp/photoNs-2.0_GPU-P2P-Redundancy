#pragma once
//#include <cuda_runtime.h>
#include "/usr/local/cuda/include/cuda_runtime.h"
//#include <vector>
//using namespace std;

////////////////////// GPU P2P DATA ///////////////////

//device arrays
extern double* d_pos_data;
extern double* d_acc_data;
extern int* d_pos_index;

extern double* d_self_pos_data;
extern double* d_self_acc_data;
extern int* d_self_pos_index;

//control memory
extern int max_res_pos_size_;
extern int max_acc_size;
extern int max_posIndexSize;
///////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

void initGPU(int verbosity_gpu);

//interactions with neighbor boxes
void getGPUMemoryState(int verbosity_gpu);
int allocMemGPU(int PROC_SIZE, int maxPartsInLeaf, int MAXTASK, int verbosity_gpu);
int copyMemGPU(double** h_pos_data, int** h_pos_index,
    int PROC_SIZE, int PROC_RANK, int maxPartsInLeaf, int numtasks, int posCounter, int verbosity_gpu);
void readResultsGPU(double** h_acc_data, int PROC_RANK, int PROC_SIZE, int maxPartsInLeaf, int MAXTASK, int partCounter, int verbosity_gpu);
int LaunchKernelP2PDualNaive(int PROC_SIZE, int PROC_RAKN, int nTasks, double SoftenScale, double MASSPART, int verbosity_gpu);
__global__ void ComputeP2PDualNaive(
                    int* d_pos_index, double* d_pos_data, double* d_acc_data, 
                    int ProcRankPosOffset, int ProcRankResOffset, int ntasks,
                    double SoftenScale, double MASSPART, int* kernelError);

//self interactions
int allocAndCopySelfInteractionsGPU(double* part_data, int* part_idx, int partDataChunk, int partIndexChunk, int resultDataChunk, int nTasks);
void LaunchKernelP2PSelfInteractions(
    int nTasks, int partDataChunk, int partIndexChunk, int resulDataChunk,
    double SoftenScale, double MASSPART);
__global__ void ComputeP2PSelfInteractions(
    int* d_self_pos_index, double* d_self_pos_data, double* dself__acc_data,
    int partDataChunk, int partIndexChunk, int resultDataChunk, int ntasks,
    double SoftenScale, double MASSPART);
void readResultsGPUSelfInteractions(double* h_acc_data, int accDataChunk, int nTasks);



#ifdef __cplusplus
}
#endif