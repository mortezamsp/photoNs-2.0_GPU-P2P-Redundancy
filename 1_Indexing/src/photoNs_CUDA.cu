#include "../inc/photoNs_CUDA.cuh"
#include "stdio.h"

//device arrays
double* d_particle_data;
int* d_leaf_data;
int* d_interaction_data;
double* d_result_data;

//control memory
int max_particle_data;
int max_leaf_data;
int max_interaction_data;
int max_result_data;

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

char gpuInitiated = 0;
void initGPU(int verbosity_gpu) 
{
    if(gpuInitiated == 1) 
        return;

    //init device
    int deviceCount;
    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) 
        //if(verbosity_gpu) 
        printf("No CUDA-capable device found!\n");
    else
        cudaSetDevice(0); // Use the first available device
    if( e!= cudaSuccess)
        //if(verbosity_gpu) 
        printf("error cudaGetDeviceCount : %s\n", cudaGetErrorString(e));

    max_particle_data = 0;
    max_leaf_data = 0;
    max_interaction_data = 0;
    max_result_data = 0;

    gpuInitiated = 1;
}

//is executed only from task 0 
int allocMemGPU(int nleafs, int maxPartsInLeaf, int maxNeighbors, int maxTasks, int verbosity_gpu)
{
    cudaError_t e;
    int upperBound = 2;
    max_particle_data = nleafs * maxPartsInLeaf * 3 * upperBound;
    max_leaf_data = nleafs * 2 * upperBound;
    max_interaction_data = nleafs * maxNeighbors * 2 * upperBound;
    max_result_data = maxTasks * maxPartsInLeaf * 3 * upperBound;

    //announce data size
    long totalSizeGPU = (max_particle_data + max_result_data) * sizeof(double) + 
                        (max_leaf_data + max_interaction_data) * sizeof(int);
    if(verbosity_gpu) printf(">> \t\tGPU data = %.1lf GB\n", (double)totalSizeGPU/1024.0/1024.0/1024.0);
    
    //allocate position data
    if(d_particle_data == NULL)
    {
        if(verbosity_gpu) printf(">> \t\treallocating d_pos_data on GPU...");
        e = cudaMalloc((void**)&d_particle_data, max_particle_data * sizeof(double)); 
        if(e != cudaSuccess)
        {
            printf("Error allocating memory for d_pos_data : ErrorCode = %d, %s\n\t\tsize = %d\n", 
                                       e, cudaGetErrorString(e), max_particle_data);
            getGPUMemoryState(verbosity_gpu);
            return -1;
        }
        if(verbosity_gpu) printf("memory allocated to d_pos_data, size = %d ...\n", max_particle_data);
    }

    //allocate leaf indices
    if(d_leaf_data == NULL)
    {
        if(verbosity_gpu) printf(">> \t\treallocating d_leaf_data on GPU...");
        e = cudaMalloc((void**)&d_leaf_data, max_leaf_data * sizeof(int)); 
        if(e != cudaSuccess)
        {
            printf("Error allocating memory for d_leaf_data : ErrorCode = %d, %s\n\t\tsize = %d\n", 
                                       e, cudaGetErrorString(e), max_leaf_data);
            getGPUMemoryState(verbosity_gpu);
            return -1;
        }
        if(verbosity_gpu) printf("memory allocated to d_leaf_data, size = %d ...\n", max_leaf_data);
    }

    //allocate interaction indices
    if(d_interaction_data == NULL)
    {
        if(verbosity_gpu) printf(">> \t\treallocating d_interaction_data on GPU...");
        e = cudaMalloc((void**)&d_interaction_data, max_interaction_data * sizeof(int)); 
        if(e != cudaSuccess)
        {
            printf("Error allocating memory for d_interaction_data : ErrorCode = %d, %s\n\t\tsize = %d\n", 
                                       e, cudaGetErrorString(e), max_interaction_data);
            getGPUMemoryState(verbosity_gpu);
            return -1;
        }
        if(verbosity_gpu) printf("memory allocated to d_interaction_data, size = %d ...\n",
                                       max_interaction_data);
    }

    //allocate result memory
    if(d_result_data == NULL)
    {
        if(verbosity_gpu) printf(">> \t\treallocating d_result_data on GPU...");
        e = cudaMalloc((void**)&d_result_data, max_result_data * sizeof(double)); 
        if(e != cudaSuccess)
        {
            printf("Error allocating memory for d_result_data : ErrorCode = %d, %s\n\t\tsize = %d\n", 
                                       e, cudaGetErrorString(e), max_result_data);
            getGPUMemoryState(verbosity_gpu);
            return -1;
        }
        if(verbosity_gpu) printf("memory allocated to d_result_data, size = %d ...\n", max_result_data);
    }
    e = cudaMemset(d_result_data, 0, max_result_data * sizeof(double));
    if(e != cudaSuccess)
    {
        printf("Error memset for d_result_data : ErrorCode = %d, %s\n\t\tsize = %d\n", 
                                    e, cudaGetErrorString(e), max_result_data);
        return -1;
    }

    return 0;
}

int copyMemGPU(double* h_pos_data, int* h_leaf_data, int* h_interactions, int numtasks, int verbosity_gpu)
{
    cudaError_t e;

    //copy position data
    if(verbosity_gpu) printf(">> \t\tcopying d_particle_data ...\n");
    e = cudaMemcpy(d_particle_data, h_pos_data, max_particle_data * sizeof(double), cudaMemcpyHostToDevice);
    if(e != cudaSuccess)
    {
        printf("\nError copy for d_particle_data : ErrorCode %d : %s\n", e, cudaGetErrorString(e));
        printf("\n\th_pos_data[0] = %f, size = %d, d_particle_data = %s\n",
                                    h_pos_data[0], max_particle_data, d_particle_data==nullptr?"null":"not null");
        return -1;
    }

    //copy leaf data
    if(verbosity_gpu) printf(">> \t\tcopying d_leaf_data ...\n");
    e = cudaMemcpy(d_leaf_data, h_leaf_data, max_leaf_data * sizeof(int), cudaMemcpyHostToDevice);
    if(e != cudaSuccess)
    {
        printf("\nError copy for d_leaf_data : ErrorCode %d : %s\n", e, cudaGetErrorString(e));
        printf("\n\th_leaf_data[0] = %d, size = %d, d_leaf_data = %s\n",
                                    h_leaf_data[0], max_leaf_data, d_leaf_data==nullptr?"null":"not null");
        return -1;
    }

    //copy interaction data
    int interaction_data = numtasks * 2;
    if(verbosity_gpu) printf(">> \t\tcopying d_interaction_data ...\n");
    e = cudaMemcpy(d_interaction_data, h_interactions, interaction_data * sizeof(int), cudaMemcpyHostToDevice);
    if(e != cudaSuccess)
    {
        printf("\nError copy for d_interaction_data : ErrorCode %d : %s\n", e, cudaGetErrorString(e));
        printf("\n\th_interactions[0] = %d, size = %d, d_interaction_data = %s\n",
                                    h_interactions[0], interaction_data, d_interaction_data==nullptr?"null":"not null");
        return -1;
    }

    return 0;
}

void getGPUMemoryState(int verbosity_gpu)
{
    size_t free_mem, total_mem;
    
    // Get memory info
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (status != cudaSuccess) {
        if(verbosity_gpu) printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(status));
    }
    
    if(verbosity_gpu) printf(">> \t\tFree memory: %zu bytes\n", free_mem);
}

void readResultsGPU(double* h_acc_data, int nTasks, int maxPartsInLeaf, int verbosity_gpu)
{
    int result_data = nTasks * maxPartsInLeaf * 3;
    cudaError_t e = cudaMemcpy(h_acc_data, d_result_data, result_data * sizeof(double), cudaMemcpyDeviceToHost);
    if(e != cudaSuccess)
    {
        //if(verbosity_gpu) 
        printf("Error copy for reading results (h_acc_data) : Error Code %d : %s\n", e, cudaGetErrorString(e));
    }
}

extern "C" {
    int LaunchKernelP2PIndexing(
        int nTasks, int posChunk, int leafChunk, int resultChunk, 
        double SoftenScale, double MASSPART, int verbosity_gpu) 
    {
        int* kernelError;
        cudaMalloc((void**)&kernelError, sizeof(int));
        cudaMemset(kernelError, 0, sizeof(int));
        
        int threadGroup = 100;
        int newTasks = nTasks/threadGroup;
        dim3 blocks(max(1,min(1024, newTasks + 1)));
        dim3 grids((int)(newTasks / blocks.x) + 1);
        if(verbosity_gpu) 
            printf(">> \t\tlaunching kernel with %d threads in groups of %d: grid.x=%d, grid.y=%d, block.x=%d\n", 
                        nTasks, threadGroup, grids.x, grids.y, blocks.x);
        ComputeP2PIndexing<<<grids, blocks>>>(
            d_particle_data, d_leaf_data, d_interaction_data, d_result_data,
            nTasks, posChunk, leafChunk, resultChunk, 
            SoftenScale, MASSPART, kernelError,
            max_particle_data, max_leaf_data, max_interaction_data, max_result_data);

        cudaError_t e = cudaDeviceSynchronize();
        if( e!= cudaSuccess)
        {
            //if(verbosity_gpu) 
            printf("error kernel ComputeP2PIndexing : %s\n", cudaGetErrorString(e));
            return -1;
        }
        int h_kernelError = 0;
        e = cudaMemcpy(&h_kernelError, kernelError, sizeof(int), cudaMemcpyDeviceToHost);
        if( e!= cudaSuccess)
        {
            //if(verbosity_gpu) 
            printf("error reading kernel state : %s\n", cudaGetErrorString(e));
            return -1;
        }
        if(h_kernelError < 0)
        {
            printf("ERROR : kernel ComputeP2PIndexing returned error %d\n", h_kernelError);
        }
        // if(h_kernelError > 0)
        // {
        //     printf("%d <particle-leaf> interactions computed\n", h_kernelError);
        // }
        return h_kernelError;
    }
}

//each thread handles interactions of a single leaf with all its neighbors
__global__ void ComputeP2PIndexing(
    double* pos_data, int* leaf_data, int* interaction_data, double* result_data, 
    int ntasks, int posChunk, int leafChunk, int resultChunk,
    double SoftenScale, double MASSPART, int* kernelError,
    int max_particle_data, int max_leaf_data, int max_interaction_data, int max_result_data) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("thread group [%d] enters...\n", tid);

    *kernelError = 0;
    //int interactions = 0;

    int oldtid = tid;
    tid = tid * 100;
    for(int tidGroup = 0; tidGroup < 100; tidGroup++)
    {
        tid++;        
        if(tid < ntasks)
        {

            // if(tid * 2 + 1 > max_interaction_data)
            //     printf("ERROR kernel: tid[old %d, new %d] interactions exceeded array: %d / %d\n",
            //                  oldtid, tid, tid * 2 + 1, max_interaction_data);

            int targetLeaf            = interaction_data[tid * 2 + 0];
            int sourceLeaf            = interaction_data[tid * 2 + 1];

            // if(targetLeaf * leafChunk > max_leaf_data)
            //     printf("ERROR kernel: tid[old %d, new %d] leaf exceeded array: %d / %d\n",
            //              oldtid, tid, targetLeaf * leafChunk, max_leaf_data);

            int numTargetParticles    = leaf_data[targetLeaf * leafChunk + 0];
            //int targetParticlesOffset = leaf_data[targetLeaf * leafChunk + 1];
            int targetParticlesOffset = targetLeaf * posChunk;
            int numSourceParticles    = leaf_data[sourceLeaf * leafChunk + 0];
            //int sourceParticlesOffset = leaf_data[sourceLeaf * leafChunk + 1];
            int sourceParticlesOffset = sourceLeaf * posChunk;

            //check errors        
            if(numTargetParticles <= 0 || numSourceParticles <= 0) 
            {
                // printf("ERROR kernel : tid[old %d, new %d] leaf[%d] numTargetParts = %d, numSourceParts = %d\n", 
                //         oldtid, tid, targetLeaf, numTargetParticles, numSourceParticles);
                // *kernelError = -1;
                //return;
                continue;
            }

            // printf("tid[%d] targetLeaf %d sourceLeaf %d numTargetParticles %d numSourceParticles %d\n", 
            //         tid,targetLeaf,sourceLeaf,numTargetParticles,numSourceParticles);

            //load source particles to registers
            double sourceLocalData[60]; //max num particles i think = 20
            int sourceOffset = 0, sourceOffsetRead = sourceParticlesOffset;
            for(int j = 0; j < numSourceParticles; j++)
            {
                // if(sourceOffsetRead+3 > max_particle_data)
                //     printf("ERROR kernel: tid[old %d, new %d] source particles exceeded array: %d / %d\n",
                //          oldtid, tid, sourceOffsetRead+3, max_particle_data);

                sourceLocalData[sourceOffset++] = pos_data[sourceOffsetRead++];
                sourceLocalData[sourceOffset++] = pos_data[sourceOffsetRead++];
                sourceLocalData[sourceOffset++] = pos_data[sourceOffsetRead++];
            }

            //printf("tid[%d] copied array to registers.\n", tid); 

            //read particles data
            int targetOffset;
            for(int i = 0; i < numTargetParticles; i++)
            {
                targetOffset = targetParticlesOffset + i*3;

                // if(targetOffset > max_particle_data)
                //     printf("ERROR kernel: target particles exceeded array: %d / %d\n", targetOffset, max_particle_data);

                double tar0 = pos_data[targetOffset + 0];
                double tar1 = pos_data[targetOffset + 1];
                double tar2 = pos_data[targetOffset + 2];

                //compute interactions with neighbors
                double res0 = 0, res1 = 0, res2 = 0;
                double dx[3], ir3, dr;
                double coeff = 2.0/sqrt(M_PI);
                //
                sourceOffset = 0; //sourceParticlesOffset;
                for(int j = 0; j < numSourceParticles; j++)
                {
                    // printf("tid[%d] <leaf %d : target %d, leaf %d : source %d> started...\n",
                    //     tid, targetLeaf, i, sourceLeaf, j); 

                    dx[0] = sourceLocalData[sourceOffset++] - tar0;
                    dx[1] = sourceLocalData[sourceOffset++] - tar1;
                    dx[2] = sourceLocalData[sourceOffset++] - tar2;

                    dr = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
                    if (dr < SoftenScale)
                        ir3 = MASSPART/(SoftenScale*SoftenScale*SoftenScale);
                    else
                        ir3 = MASSPART/(dr*dr*dr);

                    res0 +=  dx[0] * ir3;
                    res1 +=  dx[1] * ir3;
                    res2 +=  dx[2] * ir3;

                    // printf("tid[%d] <leaf %d : target %d, leaf %d : source %d> done.\n",
                    //     tid, targetLeaf, i, sourceLeaf, j); 
                }

                //update results
                //int resultOffset = leafidi * maxParticlesPerLeaf;
                int resultOffset = tid * resultChunk + i * 3;

                // if(resultOffset+3 > max_result_data)
                //     printf("ERROR kernel: results exceeded array: %d / %d\n", resultOffset+3, max_result_data);

                result_data[resultOffset + 0] = res0;
                result_data[resultOffset + 1] = res1;
                result_data[resultOffset + 2] = res2;
                
                //interactions++;
            }

            //interaction_data[tid*2] = interactions;
        }//if tid < ntask
    }//for group

    ////be sure that it works
    // __syncthreads();
    // if(oldtid == 0)
    // {
    //     interactions = 0;
    //     for(int i = 0; i < ntasks; i++)
    //         interactions += interaction_data[i*2];
    //     *kernelError = interactions;
    // }
}
