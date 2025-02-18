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

double* part_data; //each task has source and target particles, each particle has 3 position
int* part_idx; //task indices {im.npart, jm.npart, }
double* result;



void build_localtree();
void fmm_solver() ;
int intergra();
void fmm_construct( ) ;
void fmm_deconstruct( ) ;


void fmm_prepare();
void fmm_task();
void fmm_ext();

void fmm_solver_total();


#endif
