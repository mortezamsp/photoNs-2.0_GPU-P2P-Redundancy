#include "../inc/photoNs_CUDA.cuh"
#include "stdio.h"

double* d_pos_data;
double* d_acc_data;
int* d_pos_index;

double* d_self_pos_data;
double* d_self_acc_data;
int* d_self_pos_index;

int max_res_pos_size_;
int max_acc_size;
int max_posIndexSize;

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

    max_res_pos_size_ = 0;
    max_acc_size = 0;
    max_posIndexSize = 0 ;

    gpuInitiated = 1;
}

#pragma region other interactions
//is executed only from task 0 
int allocMemGPU(int PROC_SIZE, int maxPartsInLeaf, int MAXTASK, int verbosity_gpu)
{
    cudaError_t e;
    max_res_pos_size_ = maxPartsInLeaf * MAXTASK * 2 * 3;
    max_acc_size = maxPartsInLeaf * MAXTASK * 3;
    max_posIndexSize = MAXTASK * 5;

    //announce data size
    long totalSizeGPU = (max_res_pos_size_ + max_acc_size) * sizeof(double) + max_posIndexSize * sizeof(int);
    totalSizeGPU *= 2; //to reduce reallocations
    if(verbosity_gpu) printf(">> \tGPU data = %.1lf GB\n", (double)totalSizeGPU/1024.0/1024.0/1024.0);
    //getGPUMemoryState();
    
    //allocate position data
    if(d_pos_data == NULL)
    {
        if(verbosity_gpu) printf(">> \treallocating d_pos_data on GPU...");
        e = cudaMalloc((void**)&d_pos_data, PROC_SIZE * 2 * max_res_pos_size_ * sizeof(double)); //why 2 ? to prevent several reallocations and memory fragmentation
        if(e != cudaSuccess)
        {
            //if(verbosity_gpu) 
            printf("Error allocating memory for d_pos_data : ErrorCode = %d, %s\n\t\tres_pos_size_ = %d\n", 
                                       e, cudaGetErrorString(e), max_res_pos_size_*2*PROC_SIZE);
            getGPUMemoryState(verbosity_gpu);
            return -1;
        }
        if(verbosity_gpu) printf("memory allocated to d_pos_data, res_pos_size_ = %d ...\n", max_res_pos_size_);
    }


    //init result data
    //allocate
    if(d_acc_data == NULL)
    {
        if(verbosity_gpu) printf(">> \tallocating memory to d_acc_data...");
        e = cudaMalloc((void**)&d_acc_data, PROC_SIZE * max_acc_size * 2 * sizeof(double));
        if(e != cudaSuccess)
        {
            //if(verbosity_gpu) 
            printf("Error allocating memory for d_acc_data : %s\n\t\accSize = %d\n", cudaGetErrorString(e), max_acc_size);
            getGPUMemoryState(verbosity_gpu);
            return -1;
        }
        if(verbosity_gpu) printf("memory allocated to d_acc_data %1.f MB\n", (double)max_acc_size*PROC_SIZE*8*2/1024.0/1024.0 );
    }
    //memset
    if(verbosity_gpu) printf(">> \tmemsetting d_acc_data ...\n");
    e = cudaMemset(d_acc_data, 0, max_acc_size * PROC_SIZE * sizeof(double));
    if(e != cudaSuccess)
    {
        //if(verbosity_gpu) 
        printf("\nError memeset for d_acc_data : %s\n\t\trequested size = %1.f MB\n",
                                     cudaGetErrorString(e), ((double)max_acc_size * 8 * PROC_SIZE * 2 * sizeof(double))/1024./1024.0);
        getGPUMemoryState(verbosity_gpu);
        return -1;
    }

    //allocate indexing array
    if(d_pos_index == NULL)
    {
        if(verbosity_gpu) printf(">> \tallocating memory to d_pos_index...");
        e = cudaMalloc((void**)&d_pos_index, max_posIndexSize*PROC_SIZE*2*sizeof(int));
        if(e != cudaSuccess)
        {
            //if(verbosity_gpu) 
            printf("Error allocating memory for d_pos_index : %s\n\t\t4posSize = %d\n", 
                                        cudaGetErrorString(e), max_posIndexSize*PROC_SIZE);
            getGPUMemoryState(verbosity_gpu);
            return -1;
        }
        if(verbosity_gpu) printf("memory allocated to d_pos_index\n");
    }

    return 0;
}

int copyMemGPU(
    double** h_pos_data, int** h_pos_index,
    int PROC_SIZE, int PROC_RANK, int maxPartsInLeaf, int numtasks, int posCounter, int verbosity_gpu)
{
    cudaError_t e;
    int posIndexSize = numtasks * 5;

    //copy position data
    if(verbosity_gpu) printf(">> \tcopying d_pos_data ...\n");
    e = cudaMemcpy(d_pos_data+PROC_RANK*max_res_pos_size_, h_pos_data[PROC_RANK], posCounter * sizeof(double), cudaMemcpyHostToDevice);
    if(e != cudaSuccess)
    {
        //if(verbosity_gpu) 
        printf("\nError copy for d_pos_data : ErrorCode %d : %s\n", e, cudaGetErrorString(e));
        //if(verbosity_gpu) 
        printf("\n\th_pos_data[0] = %f, res_pos_size_ = %d, d_pos_data = %s\n",
                                    h_pos_data[PROC_RANK][0], posCounter, d_pos_data==nullptr?"null":"not null");
        return -1;
    }

    //copy indexing array
    if(verbosity_gpu) printf(">> \tcopying d_pos_index ...\n");
    e = cudaMemcpy(d_pos_index+ PROC_RANK*max_posIndexSize, h_pos_index[PROC_RANK], posIndexSize*sizeof(int), cudaMemcpyHostToDevice);
    if(e != cudaSuccess)
    {
        //if(verbosity_gpu) 
        printf("Error copy for d_pos_index : %s\n", cudaGetErrorString(e));
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
    
    if(verbosity_gpu) printf(">> \tFree memory: %zu bytes\n", free_mem);
}

void readResultsGPU(double** h_acc_data, int PROC_RANK, int PROC_SIZE, int maxPartsInLeaf, int MAXTASK, int partCounter, int verbosity_gpu)
{
    int max_acc_size = maxPartsInLeaf * MAXTASK * 3;
    int accSize = partCounter;
    cudaError_t e = cudaMemcpy(
            h_acc_data[PROC_RANK], 
            d_acc_data+PROC_SIZE*max_acc_size, accSize * sizeof(double), cudaMemcpyDeviceToHost);
    if(e != cudaSuccess)
    {
        //if(verbosity_gpu) 
        printf("Error copy for reading results (h_acc_data) : Error Code %d : %s\n", e, cudaGetErrorString(e));
    }
}

extern "C" {
    int LaunchKernelP2PDualNaive(int PROC_SIZE, int PROC_RAKN, int nTasks, double SoftenScale, double MASSPART, int verbosity_gpu) 
    {
        //if(verbosity_gpu) printf(">> \tSoftenScale = %f, MASSPART = %f\n", SoftenScale, MASSPART);
        int* kernelError;
        cudaMalloc((void**)&kernelError, sizeof(int));
        cudaMemset(kernelError, 0, sizeof(int));

        dim3 blocks(max(1,min(1024, nTasks)));
        dim3 grids((int)(nTasks / 1024) + 1);
        ComputeP2PDualNaive<<<grids, blocks>>>(
            d_pos_index, d_pos_data, d_acc_data,
            max_res_pos_size_ * (PROC_RAKN+1), max_acc_size * (PROC_RAKN+1), nTasks, 
            SoftenScale, MASSPART, kernelError);

        cudaError_t e = cudaDeviceSynchronize();
        if( e!= cudaSuccess)
        {
            //if(verbosity_gpu) 
            printf("error kernel ComputeP2PDualNaive : %s\n", cudaGetErrorString(e));
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
            printf("ERROR : kernel ComputeP2PDualNaive returned error %d\n", h_kernelError);
        }

        return h_kernelError;
    }
}

//each thread handles interactions of a single leaf with all its neighbors
__global__ void ComputeP2PDualNaive(
    int* d_pos_index, double* d_pos_data, double* d_acc_data, 
    int ProcRankPosOffset, int ProcRankResOffset, int ntasks,
    double SoftenScale, double MASSPART, int* kernelError) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    *kernelError = 0;
    if(tid < ntasks)
    {
        //read indexes
        int startOffset =     d_pos_index[tid * 5 + 0];// + ProcRankPosOffset;
        int leafId =          d_pos_index[tid * 5 + 1]; 
        int numParts =        d_pos_index[tid * 5 + 2];
        int numSourceParts =  d_pos_index[tid * 5 + 3];
        int resultIndex =     d_pos_index[tid * 5 + 4];// + ProcRankResOffset;


        //check errors        
        if(numParts < 0 || numSourceParts < 0) 
        {
            printf("-------------tid[%d:%d] kernel error: leaf[%d] numParts = %d, numSourceParts = %d\n", 
                    tid, ntasks, leafId, numParts, numSourceParts);
            *kernelError = -1;
            return;
        }
        if(numParts == 0 || numSourceParts == 0) 
        {
            return;
        }
        if(startOffset > ProcRankPosOffset || resultIndex > ProcRankResOffset
            || (ntasks < ntasks - 1 && d_pos_index[(tid + 1) * 5 + 0] > ProcRankPosOffset)
            || (ntasks < ntasks - 1 && d_pos_index[(tid + 1) * 5 + 4] > ProcRankPosOffset))
        {
            printf("-------------tid[%d:%d] startOffset=[%d/%d], resultIndex[%d/%d]\n", 
                    tid, ntasks, startOffset, ProcRankPosOffset, resultIndex, ProcRankResOffset);
            *kernelError = -3;
            return;
        }

        //load source data to registers
        double pos_data[30]; //maximum 10 particles per thread
        int indexOfSourceData = startOffset + numParts * 3;
        for(int i = 0; i < numSourceParts * 3; i++)
        {
            pos_data[i] = d_pos_data[indexOfSourceData + i];
        }

        //read particles data
        for(int i = 0; i < numParts; i++)
        {
            double tar0 = d_pos_data[startOffset + 0];
            double tar1 = d_pos_data[startOffset + 1];
            double tar2 = d_pos_data[startOffset + 2];
            startOffset += 3;

            //compute interactions with neighbors
            double res0 = 0, res1 = 0, res2 = 0;
            double dx[3], ir3, dr;
            double coeff = 2.0/sqrt(M_PI);
            for(int j = 0; j < numSourceParts; j++)
            {
                dx[0] = pos_data[j * 3 + 0] - tar0;
                dx[1] = pos_data[j * 3 + 1] - tar1;
                dx[2] = pos_data[j * 3 + 2] - tar2;

                dr = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
                if (dr < SoftenScale)
                    ir3 = MASSPART/(SoftenScale*SoftenScale*SoftenScale);
                else
                    ir3 = MASSPART/(dr*dr*dr);

                res0 +=  dx[0] * ir3;
                res1 +=  dx[1] * ir3;
                res2 +=  dx[2] * ir3;
            }

            //update results
            //int resultOffset = leafidi * maxParticlesPerLeaf;
            int resultOffset = resultIndex + i * 3;
            d_acc_data[resultOffset + 0] = res0;
            d_acc_data[resultOffset + 1] = res1;
            d_acc_data[resultOffset + 2] = res2;
        }
    }
}

#pragma endregion

#pragma region self-interactions
int allocAndCopySelfInteractionsGPU(
    double* part_data, int* part_idx, 
    int partDataChunk, int partIndexChunk, int resultDataChunk, int nTasks)
{
    cudaError_t e;

    //data
    if(d_self_pos_data == NULL)
    {
        e = cudaMalloc((void**)&d_self_pos_data, partDataChunk*nTasks*sizeof(double));
        if (e != cudaSuccess) 
        {
            printf("allocation GPU failed for d_self_pos_data: %s\n", cudaGetErrorString(e));
            return -1;
        }
    }
    e = cudaMemcpy(d_self_pos_data, part_data, partDataChunk*nTasks*sizeof(double), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) 
    {
        printf("memcpy GPU failed for d_self_pos_data: %s\n", cudaGetErrorString(e));
        return -1;
    }

    //indices
    if(d_self_pos_index == NULL)
    {
        e = cudaMalloc((void**)&d_self_pos_index, partIndexChunk*nTasks*sizeof(int));
        if (e != cudaSuccess) 
        {
            printf("allocation GPU failed for d_self_pos_index: %s\n", cudaGetErrorString(e));
            return -1;
        }
    }
    e = cudaMemcpy(d_self_pos_index, part_idx, partIndexChunk*nTasks*sizeof(int), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) 
    {
        printf("memcpy GPU failed for d_self_pos_index: %s\n", cudaGetErrorString(e));
        return -1;
    }
    
    //result
    if(d_self_acc_data == NULL)
    {
        e = cudaMalloc((void**)&d_self_acc_data, resultDataChunk*nTasks*sizeof(double));
        if (e != cudaSuccess) 
        {
            printf("allocation GPU failed for d_self_acc_data: %s\n", cudaGetErrorString(e));
            return -1;
        }
    }
    return 0;
}
extern "C" 
{
    void LaunchKernelP2PSelfInteractions(int nTasks, int partDataChunk, int partIndexChunk, int resulDataChunk, double SoftenScale, double MASSPART)
    {
        dim3 blocks(max(1,min(1024, nTasks)));
        dim3 grids((int)(nTasks / 1024) + 1);
        ComputeP2PSelfInteractions<<<grids, blocks>>>(
            d_self_pos_index, d_self_pos_data, d_self_acc_data,
            partDataChunk, partIndexChunk, resulDataChunk, nTasks,
            SoftenScale, MASSPART);

        cudaError_t e = cudaDeviceSynchronize();
        if( e!= cudaSuccess)
        {
            printf("ERROR : kernel ComputeP2PDualNaive : %s\n", cudaGetErrorString(e));
        }
    }
}
__global__ void ComputeP2PSelfInteractions(
    int* d_self_pos_index, double* d_self_pos_data, double* d_self_acc_data,
    int partDataChunk, int partIndexChunk, int resultDataChunk, int ntasks,
    double SoftenScale, double MASSPART)
{
    int tid = threadIdx.x * blockDim.x * blockIdx.x;

    if(tid < ntasks) 
    {
        int posidxOffset = tid * 3;
        int numtargets = d_self_pos_index[posidxOffset + 0];
        int numsources = d_self_pos_index[posidxOffset + 1];
		
        int targetOffset = tid * partDataChunk;
        int sourceOffset = targetOffset + numtargets * 3;
        int resultOffset = tid * resultDataChunk;

		double dx[3], dr, ir3;

        //load source data to registers
        double pos_data[30]; //maximum 10 particles per thread
        for(int i = 0; i < numsources * 3; i++)
        {
            pos_data[i] = d_self_pos_data[sourceOffset + i];
        }

		for (int ip=0; ip<numtargets; ip++)
		{
			double res0 = 0.0;
			double res1 = 0.0;
			double res2 = 0.0;

            int ipOffset = targetOffset + ip * 3;
			
            double px = d_self_pos_data[ipOffset + 0];
            double py = d_self_pos_data[ipOffset + 1];
            double pz = d_self_pos_data[ipOffset + 2];

			for (int jp=0; jp<numsources; jp++)
			{
				if (jp == ip)
					continue;

                //int jpOffset = sourceOffset + jp * 3;
                int jpOffset = jp * 3;

				dx[0] = pos_data[jpOffset + 0] - px;
				dx[1] = pos_data[jpOffset + 1] - py;
                dx[2] = pos_data[jpOffset + 2] - pz;

				dr = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
				if (dr < SoftenScale)
					ir3 = MASSPART/(SoftenScale*SoftenScale*SoftenScale);
				else
					ir3 = MASSPART/(dr*dr*dr);


                #ifdef LONGSHORT
				double drs = 0.5*dr/rs;
				ir3 *= (erfc(drs) + coeff*drs*exp(-drs*drs));
                #endif

				res0 +=  dx[0] * ir3;
				res1 +=  dx[1] * ir3;
				res2 +=  dx[2] * ir3;
			}

            d_self_acc_data[resultOffset + ip*3 + 0] = res0;
            d_self_acc_data[resultOffset + ip*3 + 1] = res1;
            d_self_acc_data[resultOffset + ip*3 + 2] = res2;
		}
	}
}
void readResultsGPUSelfInteractions(double* h_self_acc_data, int accDataChunk, int ntasks)
{
    cudaError_t e = cudaMemcpy(h_self_acc_data, d_self_acc_data, accDataChunk*ntasks*sizeof(double), cudaMemcpyDeviceToHost);
    if(e != cudaSuccess)
    {
        printf("memcpy error for d_self_acc_data\n");
    }
}

#pragma endregion