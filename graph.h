#ifndef BFS_CUDA_GRAPH_H
#define BFS_CUDA_GRAPH_H

#include <vector>
#include <cstdio>
#include <cstdlib>

struct Graph {
    std::vector<int> cvector; // all edges
    std::vector<int> rvector; // offset to adjacencyList for every vertex
    std::vector<int> distances; //number of edges for every vertex
    int root;
};


#endif //BFS_CUDA_GRAPH_H