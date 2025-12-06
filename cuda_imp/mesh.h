#ifndef MESH_H
#define MESH_H

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

typedef float3 Vertex;
typedef int3 Face;

__host__ __device__ inline Vertex operator+(const Vertex& a, const Vertex& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline Vertex operator*(const Vertex& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline Vertex operator*(float s, const Vertex& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// cpu side
struct HostMesh {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;

    std::vector<std::vector<int>> adjacency_list;
    std::vector<int2> unique_edges;
    std::vector<int2> edge_opposites;

    std::vector<int3> face_edge_indices;
};

// gpu side
struct DeviceMesh {
    Vertex* vertices;
    Vertex* new_vertices;
    Face* faces;

    // adjacency
    int* adj_offsets;
    int* adj_indices;
    int* valences;

    // edge data
    Vertex* odd_vertices;
    int2* edge_indices;
    int2* edge_opposites;

    // connectivity
    int3* face_edge_indices; // lookup which edges belong to this face
    Face* new_faces;         // output is new subdivided face

    int num_vertices;
    int num_faces;
    int num_adj_entries;
    int num_edges;
};

#endif
