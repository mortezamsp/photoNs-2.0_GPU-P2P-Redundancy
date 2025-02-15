#ifndef FMM_H
#define FMM_H

#include "kernels.h"
#include "operator.h"
#include "utility.h"
#include "toptree.h"
#include "remotes.h"
#include "adaptive.h"
#include "../inc/photoNs_CUDA.cuh"
#include "/usr/local/cuda/include/cuda_runtime.h"

//flat arrays for GPU
double* particle_data;
int particleChunk, particleSize;
int* leaf_data;
int leafChunk, leafSize;
double* result_data;
int resultChunk, resultSize;
int interactionChunk, interactionSize;
int* interactions_data;

//extern int cudaState;
//extern int maxNeighbors;

void build_localtree();
void fmm_solver() ;
int  intergra();
void fmm_construct( ) ;
void fmm_deconstruct( ) ;


void fmm_prepare();
void fmm_task();
void fmm_ext();

void fmm_solver_total();


#endif
