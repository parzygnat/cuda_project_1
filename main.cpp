#include <cstdio>
#include <string>

#include "graph.h"
#include "bfsCPU.h"

void runCpu(int startVertex, Graph &G) {
    G.root = startVertex;
    printf("Starting sequential bfs.\n");
    auto start = std::chrono::steady_clock::now();
    bfsCPU(startVertex, G);
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    for(int i = 0; i < G.distances.size(); i++) printf("%d ", G.distances[i]);
    printf("\n Elapsed time in milliseconds : %li ms.\n\n", duration);

}

int main(int argc, char **argv) {
    // read graph from standard input
    Graph G;
    G.cvector = {1, 3, 0, 2, 4, 4, 5, 7, 8, 6, 8};
    G.rvector = {0, 2, 5, 5, 6, 8, 9, 9, 11, 11};
    //run CPU sequential bfs
    runCpu(0, G);
    return 0;
}