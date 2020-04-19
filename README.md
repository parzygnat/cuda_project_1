# linear_gpu_graph_traversal
Implementation of the first linear time BFS algorithm on GPU with CUDA C/C++ based on "Scalable GPU Graph Traversal" by Duane Merril, Michael Garland and Andrew Grimshaw
https://mgarland.org/files/papers/gpubfs.pdf
This paper was the first to introduce a scalable approach to bringing the qudratic time complexity down to a linear one, as in traversing graphs the parallelism is not given, it's a typically sequential task, thus the solution incorporates all levels of memory management on Nvidia GPU without introducing atomicity
