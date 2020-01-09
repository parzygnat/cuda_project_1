#include <iostream>
#include <math.h>
#include "graph.h"
#include "bfscpu.h"
#include <queue>
#include <thrust/copy.h>


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

__global__ void cudabfs(int* cvector, int* rvector, int* c_queue, int* n_queue, int c_queuesize, int n_queuesize, int* block_alloc_size, int* distances, int level)
{
    int tid = threadIdx.x;
    int _initial;

    if(tid < c_queuesize) {
        __shared__ int prefixSum[1024];
        int local_tid = threadIdx.x;
        int degree = 0;
        int u = c_queue[tid];
        prefixSum[tid] = rvector[u + 1] - rvector[u];
        __shared__ int int_initial[1024];
        __shared__ int b_initial[1024];
                
        for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1) {
            __syncthreads();
            if ((local_tid & (nodeSize - 1)) == 0) {
                if (tid + (nodeSize >> 1) < c_queuesize) {
                    int nextPosition = local_tid + (nodeSize >> 1);
                    prefixSum[local_tid] += prefixSum[nextPosition];
                }
            }
        }
        if (local_tid == 0) {
            int block = tid >> 10;
            initial = block_alloc_size[block + 1] = prefixSum[local_tid];
        }
        for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
            __syncthreads();
            if ((local_tid & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int next_position = local_tid + (nodeSize >> 1);
                    int tmp = prefixSum[local_tid];
                    prefixSum[local_tid] -= prefixSum[next_position];
                    prefixSum[next_position] = tmp;

                }
            }
        }
        int iter = 0;
        for(int i = rvector[u]; i < rvector[u + 1]; i++) {
            int_initial[iter + prefixSum[tid]] = cvector[i];
            iter++;
        }
    }
    
    if(tid < initial) {
        if(int_initial[tid])
    }
        
    }
}

void runGpu(int startVertex, Graph &G) {
    G.root = startVertex;
    for (int i = 0; i < G.rvector.size() - 1; i++) G.distances.push_back(-1);
    printf("Starting cuda  bfs.\n\n\n");
    int level = 0;
    int num_blocks;
    int* c_queue;
    int* n_queue;
    int* block_alloc_size;
    int* distances;
    int* cvector;
    int* rvector;
    int c_queuesize;
    int n_queuesize;
    int num_vertices = G.rvector.size() - 1;
    cudaMallocManaged(&c_queue, num_vertices*sizeof(int));
    cudaMallocManaged(&n_queue, num_vertices*sizeof(int));
    cudaMallocManaged(&block_alloc_size, num_vertices*sizeof(int)/1024 + 1);
    cudaMallocManaged(&distances, num_vertices*sizeof(int));
    cudaMallocManaged(&cvector, G.cvector.size()*sizeof(int));
    cudaMallocManaged(&rvector, G.rvector.size()*sizeof(int));
    std::copy(G.cvector.begin(), G.cvector.end(), cvector);
    std::copy(G.rvector.begin(), G.rvector.end(), rvector);
    c_queue[0] = G.root;
    c_queuesize = 1;
    level = 0;
    n_queuesize = 0;
    auto start = std::chrono::system_clock::now();
    printf("im working\n");
    num_blocks = c_queuesize/1024 + 1;
    cudabfs<<<num_blocks, 1024>>>(cvector, rvector, c_queue, n_queue, c_queuesize, n_queuesize, block_alloc_size, distances, level);
    printf("it is indeed %d", c_queue[0]);
    c_queuesize = 0;
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\n \n\nElapsed time in milliseconds : %f ms.\n\n", duration);
    cudaFree(c_queue);
    cudaFree(n_queue);
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