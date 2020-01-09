#include <iostream>
#include <math.h>
#include "graph.h"
#include "bfscpu.h"
#include <queue>

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

__global__ void
cudabfs(int* cvector, int* rvector, int* c_queue, int* n_queue, int c_queuesize, int n_queuesize, int* block_alloc_size, int* distances, int* degrees, int level)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    printf("all good\n");
    if(tid < c_queuesize) {
        for(int i = rvector[tid]; i < rvector[tid + 1]; i++) {
            printf("\n works fine for me %d \n", cvector[tid]);
        }
    }
}

void runGpu(int startVertex, Graph &G) {
    G.root = startVertex;
    for (int i = 0; i < G.rvector.size() - 1; i++) G.distances.push_back(-1);
    printf("Starting cuda  bfs.\n\n\n");
    auto start = std::chrono::system_clock::now();
    int level = 0;
    int* c_queue;
    int* n_queue;
    int* block_alloc_size;
    int* distances;
    int* degrees;
    int* cvector;
    int* rvector;
    int c_queuesize;
    int n_queuesize;
    int num_vertices = G.rvector.size() - 1;
    cudaMallocManaged(&c_queue, num_vertices*sizeof(int));
    cudaMallocManaged(&n_queue, num_vertices*sizeof(int));
    cudaMallocManaged(&block_alloc_size, num_vertices*sizeof(int)/1024 + 1);
    cudaMallocManaged(&distances, num_vertices*sizeof(int));
    cudaMallocManaged(&degrees, num_vertices*sizeof(int));
    cudaMallocManaged(&cvector, G.cvector.size()*sizeof(int));
    cudaMallocManaged(&rvector, G.rvector.size()*sizeof(int));
    std::copy(G.cvector.begin(), G.cvector.end(), cvector);
    std::copy(G.rvector.begin(), G.rvector.end(), rvector);
    c_queue[0] = G.root;
    c_queuesize = 1;

    while(c_queuesize != 0){
        printf("im working\n")
        cudabfs<<<c_queuesize/1024 + 1, 1024>>>(cvector, rvector, c_queue, n_queue, c_queuesize, n_queuesize, block_alloc_size, distances, degrees, level);
        ++level;
        c_queuesize = 0;

    }
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\n \n\nElapsed time in milliseconds : %f ms.\n\n", duration);
    
}

__global__ void
 prescan(float *g_odata, float *g_idata, int n) 
 {
 extern __shared__ float temp[]; 
 int thid = threadIdx.x;
 int offset = 1; 
 temp[2*thid] = g_idata[2*thid]; // load input into shared memory
 temp[2*thid+1] = g_idata[2*thid+1];
 for (int d = n>>1; d > 0; d >>= 1)   // build sum in place up the tree
 {
    __syncthreads();
    if (thid < d)    
    { 
        int ai = offset*(2*thid+1)-1; 
        int bi = offset*(2*thid+2)-1;  
        temp[bi] += temp[ai];
    }
    offset <<= 2;
 } 
    if (thid == n - 1)
    {
      temp[n - 1] = 0;
    } // clear the last element  

for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
{      
    offset >>= 1;
    __syncthreads();      
    if (thid < d)      
    { 
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1; 
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;       
      } 
} 
      __syncthreads(); 
      g_odata[2*thid] = temp[2*thid]; // write results to device memory      
      g_odata[2*thid+1] = temp[2*thid+1]; 
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