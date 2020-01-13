#include <cstdio>
#include <string>

#include "graph.h"
#include "bfsCPU.h"

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
        return;
    for (int i = 0; i < G.rvector.size() - 1; i++) G.distances.push_back(-1);
    printf("Starting sequential bfs.\n\n\n");
    auto start = std::chrono::system_clock::now();
    bfsCPU(G);
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    //for (int i = G.rvector.size() - 10; i < G.rvector.size() - 1; i++) printf(" %d ", G.distances[i]);
    printf("\n \n\nElapsed time in milliseconds : %f ms.\n\n", duration);
    
}


int main(void)
{
    Graph G;
    for(int i = 1; i < 1 + 100 + 1000 + 10000; i++){
        G.cvector.push_back(i);
    }
    for(int i = 0; i < 1 + 100 + 1000 + 10000 + 1; i++) {
        if(i == 0)
        G.rvector.push_back(0);
        else if(i < 1 + 100 + 1000)
        G.rvector.push_back(100*i);
        else
        G.rvector.push_back(100 + 1000 + 10000);
    }    

    //run CPU sequential bfs
    runCpu(0, G);

    return 0;
}