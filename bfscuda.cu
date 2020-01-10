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

__global__ void expansion(int* cvector, int* rvector, int* v_queue, int* e_queue, int *v_queuesize, int* e_queuesize, int* block_alloc_size, int* distances, int level)
{
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    if(tid < *v_queuesize) {
        __shared__ int prefixSum[1024];
        int u = v_queue[tid];
        //we create a block shared array of degrees of the elements of the current vertex frontier
        prefixSum[tid] = rvector[u + 1] - rvector[u];
        
        //1s of 3 scans in this algorithm - we calculate offsets for writing ALL neighbors into a block shared array
        // blelloch exclusive scan algorithm with upsweep to the left
        for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1) {
            __syncthreads();
            if ((local_tid & (nodeSize - 1)) == 0) {
                if (tid + (nodeSize >> 1) < *v_queuesize) {
                    int nextPosition = local_tid + (nodeSize >> 1);
                    prefixSum[local_tid] += prefixSum[nextPosition];
                }
            }
        }

        if (local_tid == 0) {
            int block = tid >> 10;
            // the efect of upsweep - reduction of the whole array (number of ALL neighbors)
            *e_queuesize = block_alloc_size[block + 1] = prefixSum[local_tid];
        }
        //downsweep - now our array prefixSum has become a prefix sum of numbers of neighbors
        for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
            __syncthreads();
            if ((local_tid & (nodeSize - 1)) == 0) {
                if (tid + (nodeSize >> 1) < v_queuesize) {
                    int next_position = local_tid + (nodeSize >> 1);
                    int tmp = prefixSum[local_tid];
                    prefixSum[local_tid] -= prefixSum[next_position];
                    prefixSum[next_position] = tmp;

                }
            }
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
                    *e_queuesize = block_alloc_size[tid];
                }
                for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
                    __syncthreads();
                    if ((tid & (nodeSize - 1)) == 0) {
                        if (tid + (nodeSize >> 1) < v_queuesize) {
                            int next_position = tid + (nodeSize >> 1);
                            int tmp = block_alloc_size[tid];
                            block_alloc_size[tid] -= block_alloc_size[next_position];
                            block_alloc_size[next_position] = tmp;
                        }
                    }
                }
            }
        }

        int iter = 0;
        for(int i = rvector[u]; i < rvector[u + 1]; i++) {
            e_queue[iter + prefixSum[tid] + block_alloc_size[tid>>10]] = cvector[i];
            iter++;
        }
    }
}
__global__ void contraction(int* cvector, int* rvector, int* v_queue, int* e_queue, int *v_queuesize, int* e_queuesize, int* block_alloc_size, int* distances, int level)
{
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    __shared__ int b1_initial[*e_queuesize];
    __shared__ int b2_initial[*e_queuesize];

    if(tid < e_queuesize) {
        // we create a array of 0s and 1s signifying whether vertices in the edge frontier have already been visited
        b1_initial[local_tid] = 1;
        if(distances[e_queue[tid]] < 0)
            b1_initial[local_tid] = 0;
        // we create a copy of this and make an array with scan of the booleans. this way we will know how many valid neighbors are there to check
        b2_initial[local_tid] = b1_initial[local_tid]


        for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1) {
            __syncthreads();
            if ((local_tid & (nodeSize - 1)) == 0) {
                if (tid + (nodeSize >> 1) < _initial) {
                    int nextPosition = local_tid + (nodeSize >> 1);
                    b2_initial[local_tid] += b2_initial[nextPosition];
                }
            }
        }
        if (local_tid == 0) {
            int block = tid >> 10;
            *v_queuesize = block_alloc_size[block] = prefixSum[local_tid];
        }
        for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
            __syncthreads();
            if ((local_tid & (nodeSize - 1)) == 0) {
                if (tid + (nodeSize >> 1) < _initial) {
                    int next_position = local_tid + (nodeSize >> 1);
                    int tmp = b2_initial[local_tid];
                    b2_initial[local_tid] -= b2_initial[next_position];
                    b2_initial[next_position] = tmp;
                }
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
                    if (tid + (nodeSize >> 1) < v_queuesize) {
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
    if(b1[local_tid])
    {
        v_queue[block_alloc_size[tid>>10] + local_tid] = int1_initial[local_tid];
    }
    }


void runGpu(int startVertex, Graph &G) {
    G.root = startVertex;
    for (int i = 0; i < G.rvector.size() - 1; i++) G.distances.push_back(-1);
    printf("Starting cuda  bfs.\n\n\n");
    int level = 0;
    int num_blocks;
    int* v_queue;
    int* e_queue;
    int* block_alloc_size;
    int* distances;
    int* cvector;
    int* rvector;
    int e_queuesize;
    int e_queuesize;
    int num_vertices = G.rvector.size() - 1;
    cudaMallocManaged(&v_queue, num_vertices*sizeof(int));
    cudaMallocManaged(&e_queue, num_vertices*sizeof(int));
    cudaMallocManaged(&block_alloc_size, num_vertices*sizeof(int)/1024 + 1);
    cudaMallocManaged(&distances, num_vertices*sizeof(int));
    cudaMallocManaged(&cvector, G.cvector.size()*sizeof(int));
    cudaMallocManaged(&rvector, G.rvector.size()*sizeof(int));
    std::copy(G.cvector.begin(), G.cvector.end(), cvector);
    std::copy(G.rvector.begin(), G.rvector.end(), rvector);
    v_queue[0] = G.root;
    block_alloc_size[0] = 0;
    v_queuesize = 1;
    level = 0;
    e_queuesize = 0;
    auto start = std::chrono::system_clock::now();
    printf("im working\n");
    while(true) {
        num_blocks = v_queuesize/1024 + 1;
        expansion<<<num_blocks, 1024>>>(cvector, rvector, v_queue, e_queue, &v_queuesize, &e_queuesize, block_alloc_size, distances, level);
        num_blocks = v_queuesize/1024 + 1;
        contraction<<<num_blocks, 1024>>>(cvector, rvector, v_queue, e_queue, &v_queuesize, &e_queuesize, block_alloc_size, distances, level);
        break;
    }

    printf("the size of the new queue is %d", e_queuesize);
    v_queuesize = 0;
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    for(int i )
    printf("\n \n\nElapsed time in milliseconds : %f ms.\n\n", duration);
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
    G.cvector = {1, 3, 0, 2, 4, 4, 5, 7, 8, 6, 8};
    G.rvector = {0, 2, 5, 5, 6, 8, 9, 9, 11, 11};
    //run CPU sequential bfs
    runCpu(0, G);
    //run GPU parallel bfs
    runGpu(0, G);
    return 0;
}