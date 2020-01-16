#include <iostream>
#include <math.h>
#include "graph.h"
#include "bfscpu.h"
#include <queue>
#include <thrust/reduce.h>
#include <cooperative_groups.h>

#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4 
#define CONFLICT_FREE_OFFSET(n) \     ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

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
    //for (int i = G.rvector.size() - 10; i < G.rvector.size() - 1; i++) printf(" %d ", G.distances[i]);
    printf("\n \n\nElapsed time in milliseconds : %f ms.\n\n", duration);
    
}

__global__ void expansion(int* cvector, int* rvector, int* v_queue, int* e_queue, int *v_queuesize, int* e_queuesize, int* distances, int level, int* counter)
{
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    __shared__ int prefixSum[1024];
    __shared__ int block_alloc_size;
    int n;
    int u;
    int offset = 1;    
    prefixSum[local_tid] = 0;
    int _vsize = *v_queuesize;
    n = 1024;
    
    if(tid < _vsize) {
        u = v_queue[tid];
        prefixSum[local_tid] = rvector[u + 1] - rvector[u];
    }

    offset = 1;
    for (int d = n>>1; d > 0; d >>=1) {
        __syncthreads(); 
        if(local_tid < d && tid < _vsize) {
            int ai = offset*(2*local_tid+1)-1;
            int bi = offset*(2*local_tid+2)-1;
            //if(level == 1 && (prefixSum[ai] != 0 || prefixSum[bi] != 0)) printf("tid is %d ai is %d and bi is %d VALUE IS a: %d b: %d\n", tid, ai, bi, prefixSum[ai], prefixSum[bi]);
            prefixSum[bi] += prefixSum[ai];
        }
        offset *= 2;
    }

    if (local_tid == 0  && tid < _vsize) {
        // the efect of upsweep - reduction of the whole array (number of ALL neighbors)
        block_alloc_size = atomicAdd(counter, prefixSum[n - 1]);
        prefixSum[n - 1] = 0;
    }

    //downsweep - now our array prefixSum has become a prefix sum of numbers of neighbors
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (local_tid < d  && tid < _vsize) {
                int ai = offset*(2*local_tid+1)-1;
                int bi = offset*(2*local_tid+2)-1;
                int t = prefixSum[ai];
                prefixSum[ai] = prefixSum[bi];
                prefixSum[bi] += t;

        }
    }

    __syncthreads();

    if(tid < _vsize) 
    {
        //saving into global edge frontier buffer
        int iter = 0;
        for(int i = rvector[u]; i < rvector[u + 1]; i++) {
            e_queue[iter + prefixSum[local_tid] + block_alloc_size] = cvector[i];
            iter++;
        }

    }
}

__global__ void contraction(int* cvector, int* rvector, int* v_queue, int* e_queue, int *v_queuesize,  int* e_queuesize, int* distances, int level, int* counter)
{
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    __shared__ int b1_initial[1024];
    __shared__ int block_alloc_size;
    int n;
    int visited = 0;
    int offset = 1;
    int _esize = *e_queuesize;
    n = 1024;
    
    b1_initial[local_tid] = 0;
    if(local_tid < n && tid < _esize) {
        if(distances[e_queue[tid]] == -1)
            visited = b1_initial[local_tid] = 1;
    }
    // we create a copy of this and make an array with scan of the booleans. this way we will know how many valid neighbors are there to check
    offset = 1;
    for (int d = n>>1; d > 0; d >>=1) {
        __syncthreads();
        if(local_tid < d  && tid < _esize){
            int ai = offset*(2*local_tid+1)-1;
            int bi = offset*(2*local_tid+2)-1;
            b1_initial[bi] += b1_initial[ai];
        }
        offset *= 2; 
    }

    if (local_tid == 0  && tid < _esize) {
        // the efect of upsweep - reduction of the whole array (number of ALL neighbors)
        block_alloc_size = atomicAdd(counter, b1_initial[n - 1]);
        b1_initial[n - 1] = 0;
    }
    //downsweep - now our array prefixSum has become a prefix sum of numbers of neighbors
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (local_tid < d && tid < _esize) {
            int ai = offset*(2*local_tid+1)-1;
            int bi = offset*(2*local_tid+2)-1;
            int t = b1_initial[ai];
            b1_initial[ai] = b1_initial[bi];
            b1_initial[bi] += t;
        }

    }

    __syncthreads();
    //now we compact
    if(tid < _esize && visited)
    {
        int u = e_queue[tid];
        distances[u] = level + 1;
        if(local_tid==0) {
        }
        v_queue[b1_initial[local_tid] + block_alloc_size] = u;
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
    int* rvector;
    int *e_queuesize;
    int *v_queuesize;
    int num_vertices = G.rvector.size() - 1;
    int num_edges = G.cvector.size();

    //cuda unified memory allocations
    cudaMallocManaged(&e_queuesize, sizeof(int));
    cudaMallocManaged(&v_queuesize, sizeof(int));
    cudaMallocManaged(&counter, sizeof(int));
    cudaMallocManaged(&v_queue, num_edges*sizeof(int));
    cudaMallocManaged(&e_queue, num_edges*sizeof(int));
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
    num_threads = 1024;
    *counter = 0;
    *e_queuesize = 0;
    printf("Starting cuda  bfs.\n\n\n");
    auto start = std::chrono::system_clock::now();
    while(*v_queuesize) { // it will work until the size of vertex frontier is 0
        *counter = 0;
        num_blocks = (*v_queuesize)/1024 + 1;
        //1st phase -> we expand vertex frontier into edge frontier by copying ALL possible neighbors
        //no threads stay idle apart from last block if num_threads > 1024, all SIMD lanes are utilized when reading from global memory
        expansion<<<num_blocks, num_threads>>>(cvector, rvector, v_queue, e_queue, v_queuesize, e_queuesize, distances, level, counter);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        *e_queuesize = *counter;
        if(*e_queuesize == 0) break;
        //printf("E SIZE: %d\n", *e_queuesize);
        *counter = 0;
        num_blocks = (*e_queuesize)/1024 + 1;
        contraction<<<num_blocks, num_threads>>>(cvector, rvector, v_queue, e_queue, v_queuesize, e_queuesize, distances, level, counter);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        *v_queuesize = *counter;
        //printf("V SIZE: %d\n", *v_queuesize);
        level++;
    }
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\n \n\nElapsed time in milliseconds : %f ms.\n\n", duration);
    //for (int i = 0; i < 8; i++) printf(" %d ", distances[i]);
    //for (int i = G.rvector.size() - 10; i < G.rvector.size() - 1; i++) printf(" %d ", distances[i]);
    cudaFree(v_queuesize);
    cudaFree(e_queuesize);
    cudaFree(v_queue);
    cudaFree(counter);
    cudaFree(e_queue);
    cudaFree(distances);
    cudaFree(cvector);
    cudaFree(rvector);
    
}


int main(int argc, char *argv[])
{
    if(argc < 2)
  { 
    printf("Not enough arguments\n");
   return 0;
  }

    int config = atoi(argv[1]);
    Graph G;
    for(int i = 1; i < 1 + config + config*config + config*config*config; i++){
        G.cvector.push_back(i);
    }
    for(int i = 0; i < 1 + config + config*config + config*config*config + 1; i++) {
        if(i == 0)
        G.rvector.push_back(0);
        else if(i < 1 + config + config*config)
        G.rvector.push_back(config*i);
        else
        G.rvector.push_back(config*config*config + config*config + config);
    }

    // G.cvector = {1, 3, 0, 2 , 4, 4, 5, 7, 8, 6, 8};
    // G.rvector = {0, 2, 5, 5, 6, 8, 9, 9, 11, 11};

    //run GPU parallel bfs
    runGpu(0, G);
    

    //run CPU sequential bfs
    runCpu(0, G);

    return 0;
}