#include <iostream>
#include <math.h>
#include "graph.h"
#include "bfsCPU.h"
#include <queue>

void bfsCPU(Graph &G) {
    for (int i = 0; i < G.rvector.size() - 1; i++) G.distances.push_back(-1);
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
    printf("Starting sequential bfs.\n");


    auto start = std::chrono::system_clock::now();
    bfsCPU(G);
    auto end = std::chrono::system_clock::now();


    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    for(int i = 0; i < G.distances.size(); i++) printf("%d ", G.distances[i]);

    printf("\n Elapsed time in milliseconds : %f ms.\n\n", duration);
    
}

int main(void)
{
    Graph G;
    G.cvector = {1, 3, 0, 2, 4, 4, 5, 7, 8, 6, 8};
    G.rvector = {0, 2, 5, 5, 6, 8, 9, 9, 11, 11};
    //run CPU sequential bfs
    runCpu(0, G);
    return 0;
}



extern "C" {

__global__
void simpleBfs(int N, int level, int *d_adjacencyList, int *d_edgesOffset,
               int *d_edgesSize, int *d_distance, int *d_parent, int *changed) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;

    if (thid < N && d_distance[thid] == level) {
        int u = thid;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (level + 1 < d_distance[v]) {
                d_distance[v] = level + 1;
                d_parent[v] = i;
                valueChange = 1;
            }
        }
    }

    if (valueChange) {
        *changed = valueChange;
    }
}

__global__
void queueBfs(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
              int queueSize, int *nextQueueSize, int *d_currentQueue, int *d_nextQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[v] == INT_MAX && atomicMin(&d_distance[v], level + 1) == INT_MAX) {
                d_parent[v] = i;
                int position = atomicAdd(nextQueueSize, 1);
                d_nextQueue[position] = v;
            }
        }
    }
}

//Scan bfs
__global__
void nextLayer(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
               int queueSize, int *d_currentQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (level + 1 < d_distance[v]) {
                d_distance[v] = level + 1;
                d_parent[v] = i;
            }
        }
    }
}

__global__
void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_parent,
                  int queueSize, int *d_currentQueue, int *d_degrees) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        int degree = 0;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != u) {
                ++degree;
            }
        }
        d_degrees[thid] = degree;
    }
}

__global__
void scanDegrees(int size, int *d_degrees, int *incrDegrees) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < size) {
        //write initial values to shared memory
        __shared__ int prefixSum[1024];
        int modulo = threadIdx.x;
        prefixSum[modulo] = d_degrees[thid];
        __syncthreads();

        //calculate scan on this block
        //go up
        for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int nextPosition = modulo + (nodeSize >> 1);
                    prefixSum[modulo] += prefixSum[nextPosition];
                }
            }
            __syncthreads();
        }

        //write information for increment prefix sums
        if (modulo == 0) {
            int block = thid >> 10;
            incrDegrees[block + 1] = prefixSum[modulo];
        }

        //go down
        for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int next_position = modulo + (nodeSize >> 1);
                    int tmp = prefixSum[modulo];
                    prefixSum[modulo] -= prefixSum[next_position];
                    prefixSum[next_position] = tmp;

                }
            }
            __syncthreads();
        }
        d_degrees[thid] = prefixSum[modulo];
    }

}

__global__
void assignVerticesNextQueue(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_parent, int queueSize,
                             int *d_currentQueue, int *d_nextQueue, int *d_degrees, int *incrDegrees,
                             int nextQueueSize) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        __shared__ int sharedIncrement;
        if (!threadIdx.x) {
            sharedIncrement = incrDegrees[thid >> 10];
        }
        __syncthreads();

        int sum = 0;
        if (threadIdx.x) {
            sum = d_degrees[thid - 1];
        }

        int u = d_currentQueue[thid];
        int counter = 0;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != u) {
                int nextQueuePlace = sharedIncrement + sum + counter;
                d_nextQueue[nextQueuePlace] = v;
                counter++;
            }
        }
    }
}

}
