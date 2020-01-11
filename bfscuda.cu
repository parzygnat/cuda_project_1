#include <iostream>
#include <math.h>
#include "graph.h"
#include "bfscpu.h"
#include <queue>
#include <thrust/reduce.h>


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

__global__ void expansion(int* cvector, int* rvector, int* v_queue, int* e_queue, int *v_queuesize, int* e_queuesize, int* block_alloc_size, int* distances, int level, int extra)
{
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    __shared__ int prefixSum[1024];
    int u = v_queue[tid];
    int n = *v_queuesize;
    
    if(tid < extra) {

        if(*v_queuesize > 1024) {
            n = 1024;
        }
    }
    
        if(tid < extra && tid >= n) {
            prefixSum[tid] = 0;
        }
    

        if(tid < n) {
        //we create a block shared array of degrees of the elements of the current vertex frontier
        prefixSum[tid] = rvector[u + 1] - rvector[u];
        
        }
        
        if(tid < extra) {
        //1s of 3 scans in this algorithm - we calculate offsets for writing ALL neighbors into a block shared array
        // blelloch exclusive scan algorithm with upsweep to the left
        int offset = 1;
        for (int d = extra>>1; d > 0; d >>=1) {
            __syncthreads();
                    if(local_tid < d)
                    {
                    int ai = offset*(2*tid+1)-1;
                    int bi = offset*(2*tid+2)-1;
                    prefixSum[bi] += prefixSum[ai];
                    }
                    offset *= 2;
                
            
        }

        if (local_tid == 0) {
            int block = tid >> 10;
            // the efect of upsweep - reduction of the whole array (number of ALL neighbors)
            e_queuesize[0] = block_alloc_size[block + 1] = prefixSum[extra - 1];
            prefixSum[extra - 1] = 0;
            *v_queuesize = 0;

        }
        //downsweep - now our array prefixSum has become a prefix sum of numbers of neighbors
        for (int d = 1; d < extra; d *= 2) {
            offset >>= 1;
            __syncthreads();
            if (local_tid < d) {
                    int ai = offset*(2*tid+1)-1;
                    int bi = offset*(2*tid+2)-1;

                    int t = prefixSum[ai];
                    prefixSum[ai] = prefixSum[bi];
                    prefixSum[bi] += t;

            }
        }

        


        //scan on offsets produced by blocks in total
        if(gridDim.x > 1) {
            if(tid < gridDim.x) {
                for (int nodeSize = 2; nodeSize <= gridDim.x; nodeSize <<= 1) {
                    __syncthreads();
                    if ((tid & (nodeSize - 1)) == 0) {
                            int nextPosition = tid + (nodeSize >> 1);
                            block_alloc_size[tid] += block_alloc_size[nextPosition];
                        }
                    
                }
                if (tid == 0) {
                    *e_queuesize = block_alloc_size[tid];
                }
                for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
                    __syncthreads();
                    if ((tid & (nodeSize - 1)) == 0) {
                            int next_position = tid + (nodeSize >> 1);
                            int tmp = block_alloc_size[tid];
                            block_alloc_size[tid] -= block_alloc_size[next_position];
                            block_alloc_size[next_position] = tmp;
                        }
                    
                }
            }
        }
        int iter = 0;
        int temp = block_alloc_size[tid>>10];
        if (gridDim.x == 1) temp = 0;
        for(int i = rvector[u]; i < rvector[u + 1]; i++) {
            e_queue[iter + prefixSum[local_tid] + temp] = cvector[i];
            iter++;
        }

    }
}
__global__ void contraction(int* cvector, int* rvector, int* v_queue, int* e_queue, int *v_queuesize, int* e_queuesize, int* block_alloc_size, int* distances, int level, int extra)
{

    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    extern __shared__ int array[];
    int* b1_initial = (int*)array; 
    int* b2_initial = b1_initial + *e_queuesize;
    int n = *e_queuesize;
    if(*e_queuesize > 1024) {
        n = 1024;
    }

    if(tid < extra && tid >= n) {
        b1_initial[tid] = 0;
    }

    if(local_tid < n) {
        b1_initial[local_tid] = 1;
        if(distances[e_queue[tid]] >= 0)
            b1_initial[local_tid] = 0;
    }

    if(tid < extra) {

        // we create a copy of this and make an array with scan of the booleans. this way we will know how many valid neighbors are there to check
        b2_initial[local_tid] = b1_initial[local_tid];

        int offset = 1;
        for (int d = extra>>1; d > 0; d >>=1) {
            __syncthreads();
                    if(local_tid < d)
                    {
                    int ai = offset*(2*tid+1)-1;
                    int bi = offset*(2*tid+2)-1;
                    b2_initial[bi] += b2_initial[ai];
                    }
                    offset *= 2;
                
            
        }

        if (local_tid == 0) {
            int block = tid >> 10;
            // the efect of upsweep - reduction of the whole array (number of ALL neighbors)
            v_queuesize[0] = block_alloc_size[block] = b2_initial[extra - 1];
            b2_initial[extra - 1] = 0;

        }
        //downsweep - now our array prefixSum has become a prefix sum of numbers of neighbors
        for (int d = 1; d < extra; d *= 2) {
            offset >>= 1;
            __syncthreads();
            if (local_tid < d) {
                    int ai = offset*(2*tid+1)-1;
                    int bi = offset*(2*tid+2)-1;

                    int t = b2_initial[ai];
                    b2_initial[ai] = b2_initial[bi];
                    b2_initial[bi] += t;

                
            }
        }
        __syncthreads();
        // now we have an array of neighbors, a mask signifying which we can copy further, and total number of elements to copy
    }

    //scan on offsets produced by blocks in total
    if(gridDim.x > 1) {
        if(tid < gridDim.x) {
            for (int nodeSize = 2; nodeSize <= gridDim.x; nodeSize <<= 1) {
                __syncthreads();
                if ((tid & (nodeSize - 1)) == 0) {
                    if (tid + (nodeSize >> 1) < gridDim.x) {
                        int nextPosition = tid + (nodeSize >> 1);
                        block_alloc_size[tid] += block_alloc_size[nextPosition];
                    }
                }
            }
            if (tid == 0) {
                *v_queuesize = block_alloc_size[tid];
            }
            for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
                __syncthreads();
                if ((tid & (nodeSize - 1)) == 0) {
                    if (tid + (nodeSize >> 1) < *v_queuesize) {
                        int next_position = tid + (nodeSize >> 1);
                        int tmp = block_alloc_size[tid];
                        block_alloc_size[tid] -= block_alloc_size[next_position];
                        block_alloc_size[next_position] = tmp;
                    }
                }
            }
        }
    }
    
    //now we compact
    if(b1_initial[local_tid])
    {
        int temp = block_alloc_size[tid>>10];
        if (gridDim.x == 1) temp = 0;
        distances[e_queue[local_tid]] = level + 1;
        v_queue[temp + b2_initial[local_tid]] = e_queue[local_tid];
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
    int* block_alloc_size;
    int* distances;
    int* cvector;
    int* rvector;
    int *e_queuesize;
    int *v_queuesize;
    int num_vertices = G.rvector.size() - 1;

    //cuda unified memory allocations
    cudaMallocManaged(&e_queuesize, sizeof(int));
    cudaMallocManaged(&v_queuesize, sizeof(int));
    cudaMallocManaged(&v_queue, num_vertices*sizeof(int));
    cudaMallocManaged(&e_queue, num_vertices*sizeof(int));
    cudaMallocManaged(&block_alloc_size, num_vertices*sizeof(int)/1024 + 1);
    cudaMallocManaged(&distances, num_vertices*sizeof(int));
    
    //initializations 
    memset(distances, -1, num_vertices*sizeof(int));
    distances[G.root] = 0; 
    cudaMallocManaged(&cvector, G.cvector.size()*sizeof(int));
    cudaMallocManaged(&rvector, G.rvector.size()*sizeof(int));
    std::copy(G.cvector.begin(), G.cvector.end(), cvector);
    std::copy(G.rvector.begin(), G.rvector.end(), rvector);
    v_queue[0] = G.root;
    //for (int i = 0; i < G.rvector.size() - 1; i++) G.distances.push_back(-1);
    block_alloc_size[0] = 0;
    *v_queuesize = 1;
    level = 0;
    int mem;
    int extra;
    *e_queuesize = 0;
    printf("Starting cuda  bfs.\n\n\n");
    auto start = std::chrono::system_clock::now();
    while(*v_queuesize) { // it will work until the size of vertex frontier is 0
        extra = *v_queuesize;
        extra--;
        extra |= extra >> 1;
        extra |= extra >> 2;
        extra |= extra >> 4;
        extra |= extra >> 8;
        extra |= extra >> 16;
        extra++;
        //number of blocks scaled to the frontier size
        num_blocks = extra/1024 + 1;
        //if queue size is bigger than 1024 the numbers of threads has to be kept at 1024 because it's the maximum on CUDA
        if(num_blocks==1) num_threads = extra; else num_threads = 1024;
        //1st phase -> we expand vertex frontier into edge frontier by copying ALL possible neighbors
        //no threads stay idle apart from last block if num_threads > 1024, all SIMD lanes are utilized when reading from global memory

        expansion<<<num_blocks, num_threads>>>(cvector, rvector, v_queue, e_queue, v_queuesize, e_queuesize, block_alloc_size, distances, level, extra);
        cudaDeviceSynchronize();
        extra = *e_queuesize;
        extra--;
        extra |= extra >> 1;
        extra |= extra >> 2;
        extra |= extra >> 4;
        extra |= extra >> 8;
        extra |= extra >> 16;
        extra++;
        //print newly produced edge frontier
        printf("E: size: %d, [", *e_queuesize); for(int i = 0; i < *e_queuesize; i++) printf("%d ", e_queue[i]); printf("]\n");
        num_blocks = (extra)/1024 + 1;
        if(num_blocks==1) num_threads = extra; else num_threads = 1024;
        mem = (extra)*2*sizeof(int);
        contraction<<<num_blocks, num_threads, mem>>>(cvector, rvector, v_queue, e_queue, v_queuesize, e_queuesize, block_alloc_size, distances, level, extra);
        cudaDeviceSynchronize();
        printf("V: size: %d, [", *v_queuesize); for(int i = 0; i < *v_queuesize; i++) printf("%d ", v_queue[i]); printf("]\n");
        level++;
    }
    distances[G.root] = 0;
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    for(int i = 0; i < num_vertices; i++) printf("%d ", distances[i]);
    printf("\n \n\nElapsed time in milliseconds : %f ms.\n\n", duration);
    cudaFree(v_queuesize);
    cudaFree(e_queuesize);
    cudaFree(v_queue);
    cudaFree(e_queue);
    cudaFree(block_alloc_size);
    cudaFree(distances);
    cudaFree(cvector);
    cudaFree(rvector);
    
}


int main(void)
{
    Graph G;
    for(int i = 1; i < 993; i++){
        G.cvector.push_back(i);
    }
    for(int i = 0; i < 993; i++) {
        if(i < 32)
        G.rvector.push_back(i*31);
        else
        G.rvector.push_back(31*32);
    }
    //run CPU sequential bfs
    runCpu(0, G);
    //run GPU parallel bfs
    runGpu(0, G);
    return 0;
}