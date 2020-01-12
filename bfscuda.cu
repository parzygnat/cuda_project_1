#include <iostream>
#include <math.h>
#include "graph.h"
#include "bfscpu.h"
#include <queue>
#include <thrust/reduce.h>
#include <cooperative_groups.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void bfsCPU(Graph &G) {
    G.distances[G.root] = 0;
    std::queue<int> Q;
    Q.push(G.root);

    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();

        for (int i = G.rvector[u]; i < G.rvector[u + 1]; i++) {
            int v = G.cvector[i];
            if (G.distances[v] == -1) {
                G.distances[v] = G.distances[u] + 1;
                Q.push(v);
            }
        }
    }
}

void runCpu(int startVertex, Graph &G) {
    G.root = startVertex;
    for (int i = 0; i < G.rvector.size() - 1; i++) G.distances.push_back(-1);
    printf("Starting sequential bfs.\n\n\n");
    auto start = std::chrono::system_clock::now();
    bfsCPU(G);
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    for(int i = 0; i < G.distances.size(); i++) printf("%d ", G.distances[i]);
    printf("\n \n\nElapsed time in milliseconds : %f ms.\n\n", duration);
    
}

__global__ void expansion(int* cvector, int* rvector, int* v_queue, int* e_queue, int *v_queuesize, int* e_queuesize, int* distances, int level, int extra, int* counter)
{
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    __shared__ int prefixSum[1024];
    __shared__ int total;
    __shared__ int block_alloc_size;
    int u = v_queue[tid];
    int n = *v_queuesize;
    int offset = 1;
    
    
    if(tid < extra) {
        if(*v_queuesize > 1024) {
            n = 1024;
        }
        else n = extra;
    }
    
    if(tid < extra && tid >= *v_queuesize) {
        prefixSum[local_tid] = 0;
    }
    if(level == 1) return;

    if(tid < n) {
    //we create a block shared array of degrees of the elements of the current vertex frontier
        prefixSum[local_tid] = rvector[u + 1] - rvector[u];
        //if(level == 1) printf("my tid is %d and my prefix sum is %d u is %d and u+1 is %d\n", tid, prefixSum[local_tid], rvector[u], rvector[u + 1]);
    }
    
    //1s of 4 scans in this algorithm - we calculate offsets for writing ALL neighbors into a block shared array
    // blelloch exclusive scan algorithm with upsweep to the left
        for (int d = n>>1; d > 0; d >>=1) {
            __syncthreads();
                    if(local_tid < d && tid < extra)
                    {
                    int ai = offset*(2*local_tid+1)-1;
                    int bi = offset*(2*local_tid+2)-1;
                    prefixSum[bi] += prefixSum[ai];
                    }
                    offset *= 2;
                
            
        }

        if (local_tid == 0  && tid < extra) {
            // the efect of upsweep - reduction of the whole array (number of ALL neighbors)
            total = prefixSum[n - 1];
            prefixSum[n - 1] = 0;
        }
        //downsweep - now our array prefixSum has become a prefix sum of numbers of neighbors
        for (int d = 1; d < n; d *= 2) {
            offset >>= 1;
            __syncthreads();
            if (local_tid < d  && tid < extra) {
                    int ai = offset*(2*local_tid+1)-1;
                    int bi = offset*(2*local_tid+2)-1;

                    int t = prefixSum[ai];
                    prefixSum[ai] = prefixSum[bi];
                    prefixSum[bi] += t;

            }
        }

    __syncthreads();

    if(local_tid == 0) {
        block_alloc_size = atomicAdd(counter, total);
    }


    if(tid < *v_queuesize) {
    //saving into global edge frontier buffer
    int iter = 0;
    int temp = block_alloc_size;
    if (gridDim.x == 1) temp = 0;
    for(int i = rvector[u]; i < rvector[u + 1]; i++) {
        e_queue[iter + prefixSum[local_tid] + temp] = cvector[i];
        iter++;
    }

}
}

__global__ void contraction(int* cvector, int* rvector, int* v_queue, int* e_queue, int *v_queuesize,  int* e_queuesize, int* distances, int level, int extra, int* counter)
{
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    extern __shared__ int b1_initial[];
    __shared__ int block_alloc_size;
    __shared__ int total;
    int n;
    int offset = 1;

    if(tid < extra) {
        if(*e_queuesize > 1024) {
            n = 1024;
        }
        else n = extra;
    }

    if(tid < extra && tid >= *e_queuesize) {
        b1_initial[local_tid] = 0;
    }

    
    if(local_tid < n && tid < *e_queuesize) {
        b1_initial[local_tid] = 1;

        if(distances[e_queue[tid]] >= 0)
            b1_initial[local_tid] = 0;
    }


    // we create a copy of this and make an array with scan of the booleans. this way we will know how many valid neighbors are there to check

        offset = 1;
        for (int d = n>>1; d > 0; d >>=1) {
            __syncthreads();
                    if(local_tid < d  && tid < extra)
                    {
                    int ai = offset*(2*local_tid+1)-1;
                    int bi = offset*(2*local_tid+2)-1;
                    b1_initial[bi] += b1_initial[ai];
                    }
                    offset *= 2;
                
            
        }

        if (local_tid == 0  && tid < extra) {
            // the efect of upsweep - reduction of the whole array (number of ALL neighbors)
            *v_queuesize = b1_initial[n - 1];
            total = *v_queuesize;
            //printf("\n i, thread no %d, im setting index %d of block_offsets to %d\n", tid, block, b1_initial[n - 1]);
            b1_initial[n - 1] = 0;

        }


        //downsweep - now our array prefixSum has become a prefix sum of numbers of neighbors
        for (int d = 1; d < n; d *= 2) {
            offset >>= 1;
            __syncthreads();
            if (local_tid < d && tid < extra) {
                    int ai = offset*(2*local_tid+1)-1;
                    int bi = offset*(2*local_tid+2)-1;

                    int t = b1_initial[ai];
                    b1_initial[ai] = b1_initial[bi];
                    b1_initial[bi] += t;

                
            }
        }

        __syncthreads();

    if(local_tid == 0) {
        block_alloc_size = atomicAdd(counter, total);
    }
    __syncthreads();

    if(local_tid == 1023 || tid == *e_queuesize) {
        if(distances[e_queue[tid]] >= 0)
            return;
        int ver = e_queue[tid];
        int temp = block_alloc_size;
        if (gridDim.x == 1) temp = 0;
        distances[ver] = level + 1;
        v_queue[temp + b1_initial[local_tid]] = ver;
        return;
    }
    //now we compact
    if(tid < *e_queuesize && (b1_initial[local_tid] != b1_initial[local_tid + 1]))
    {
        int ver = e_queue[tid];
        int temp = block_alloc_size;
        if (gridDim.x == 1) temp = 0;
        distances[ver] = level + 1;
        if(local_tid==0) {
        }
        v_queue[b1_initial[local_tid] + temp] = ver;
    }
}
    


void runGpu(int startVertex, Graph &G) {
    //declarations 
    G.root = startVertex;
    int level = 0;
    int num_blocks;
    int num_threads;
    int* v_queue;
    int* e_queue;
    int* counter;
    int* distances;
    int* cvector;
    int* extra;
    int* rvector;
    int *e_queuesize;
    int *v_queuesize;
    int num_vertices = G.rvector.size() - 1;

    //cuda unified memory allocations
    cudaMallocManaged(&e_queuesize, sizeof(int));
    cudaMallocManaged(&v_queuesize, sizeof(int));
    cudaMallocManaged(&counter, sizeof(int));
    cudaMallocManaged(&extra, sizeof(int));
    cudaMallocManaged(&v_queue, num_vertices*sizeof(int));
    cudaMallocManaged(&e_queue, num_vertices*sizeof(int));
    cudaMallocManaged(&distances, num_vertices*sizeof(int));
    
    //initializations 
    memset(distances, -1, num_vertices*sizeof(int));
    distances[G.root] = 0; 
    cudaMallocManaged(&cvector, G.cvector.size()*sizeof(int));
    cudaMallocManaged(&rvector, G.rvector.size()*sizeof(int));
    std::copy(G.cvector.begin(), G.cvector.end(), cvector);
    std::copy(G.rvector.begin(), G.rvector.end(), rvector);
    v_queue[0] = G.root;
    *v_queuesize = 1;
    level = 0;
    int mem;
    *counter = 0;
    *e_queuesize = 0;
    printf("Starting cuda  bfs.\n\n\n");
    auto start = std::chrono::system_clock::now();
    while(*v_queuesize) { // it will work until the size of vertex frontier is 0
        *counter = 0;
        *extra = *v_queuesize;
        *extra--;
        *extra |= *extra >> 1;
        *extra |= *extra >> 2;
        *extra |= *extra >> 4;
        *extra |= *extra >> 8;
        *extra |= *extra >> 16;
        *extra++;
        //number of blocks scaled to the frontier size
        num_blocks = *extra/1025 + 1;
        //if queue size is bigger than 1024 the numbers of threads has to be kept at 1024 because it's the maximum on CUDA
        if(num_blocks==1) num_threads = *extra; else num_threads = 1024;
        //1st phase -> we expand vertex frontier into edge frontier by copying ALL possible neighbors
        //no threads stay idle apart from last block if num_threads > 1024, all SIMD lanes are utilized when reading from global memory
        expansion<<<num_blocks, num_threads>>>(cvector, rvector, v_queue, e_queue, v_queuesize, e_queuesize, distances, level, *extra, counter);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        *e_queuesize = *counter;
        *counter = 0;
        *extra = *e_queuesize;
        *extra--;
        *extra |= *extra >> 1;
        *extra |= *extra >> 2;
        *extra |= *extra >> 4;
        *extra |= *extra >> 8;
        *extra |= *extra >> 16;
        *extra++;

        num_blocks = (*extra)/1025 + 1;
        if(num_blocks==1) num_threads = *extra; else num_threads = 1024;
        mem = (num_threads)*sizeof(int);
        if(mem == 0) break;
        contraction<<<num_blocks, num_threads, mem>>>(cvector, rvector, v_queue, e_queue, v_queuesize, e_queuesize, distances, level, *extra, counter);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        *v_queuesize = *counter;
        level++;
    }
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\n \n\nElapsed time in milliseconds : %f ms.\n\n", duration);
    cudaFree(v_queuesize);
    cudaFree(e_queuesize);
    cudaFree(v_queue);
    cudaFree(e_queue);
    cudaFree(distances);
    cudaFree(cvector);
    cudaFree(rvector);
    
}


int main(void)
{
    Graph G;
    for(int i = 1; i < 1 + 10000; i++){
        G.cvector.push_back(i);
    }
    for(int i = 0; i < 1 + 10000 + 1; i++) {
        if(i == 0)
        G.rvector.push_back(0);
        else
        G.rvector.push_back(10000);
    }
    //run CPU sequential bfs
    runCpu(0, G);
    //run GPU parallel bfs
    runGpu(0, G);
    return 0;
}