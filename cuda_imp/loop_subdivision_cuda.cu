#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "mesh.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// helper structs

struct EdgeInfo {
    int face_idx;
    int edge_order;
    int opposite_v;
};

struct EdgeKeyCompare {
    __host__ __device__ bool operator()(const int2& a, const int2& b) const {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    }
};

struct DirectedEdgeSort {
    __host__ __device__ bool operator()(const int2& a, const int2& b) const {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    }
};

struct CompareInt2ToInt {
    __host__ __device__ bool operator()(const int2& edge, const int& val) const {
        return edge.x < val;
    }
};

// subdivision kernels

__global__ void reposition_even_kernel(DeviceMesh mesh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= mesh.num_vertices) return;

    Vertex v_curr = mesh.vertices[idx];
    int n = mesh.valences[idx];

    if (n == 0) {
        mesh.new_vertices[idx] = v_curr;
        return;
    }

    int start_idx = mesh.adj_offsets[idx];
    Vertex sum_neighbors = make_float3(0.0f, 0.0f, 0.0f);

    for (int k = 0; k < n; ++k) {
        int neighbor_idx = mesh.adj_indices[start_idx + k];
        sum_neighbors = sum_neighbors + mesh.vertices[neighbor_idx];
    }

    float beta;
    if (n == 3) beta = 3.0f / 16.0f;
    else {
        float term = 3.0f / 8.0f + 0.25f * cosf(2.0f * M_PI / n);
        beta = (1.0f / n) * (5.0f / 8.0f - term * term);
    }

    float alpha = 1.0f - (n * beta);
    mesh.new_vertices[idx] = (v_curr * alpha) + (sum_neighbors * beta);
}

__global__ void compute_odd_vertices_kernel(DeviceMesh mesh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= mesh.num_edges) return;

    int2 edge = mesh.edge_indices[idx];
    Vertex v1 = mesh.vertices[edge.x];
    Vertex v2 = mesh.vertices[edge.y];
    int2 opp = mesh.edge_opposites[idx];

    Vertex v_odd;
    if (opp.x != -1 && opp.y != -1) {
        Vertex v_opp1 = mesh.vertices[opp.x];
        Vertex v_opp2 = mesh.vertices[opp.y];
        v_odd = (v1 + v2) * (3.0f / 8.0f) + (v_opp1 + v_opp2) * (1.0f / 8.0f);
    } else {
        v_odd = (v1 + v2) * 0.5f;
    }
    mesh.odd_vertices[idx] = v_odd;
}

__global__ void rebuild_faces_kernel(DeviceMesh mesh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= mesh.num_faces) return;

    Face old_face = mesh.faces[idx];
    int v0 = old_face.x; int v1 = old_face.y; int v2 = old_face.z;
    int3 edge_idxs = mesh.face_edge_indices[idx];

    int e01 = mesh.num_vertices + edge_idxs.x;
    int e12 = mesh.num_vertices + edge_idxs.y;
    int e20 = mesh.num_vertices + edge_idxs.z;

    mesh.new_faces[4 * idx + 0] = make_int3(v0, e01, e20);
    mesh.new_faces[4 * idx + 1] = make_int3(v1, e12, e01);
    mesh.new_faces[4 * idx + 2] = make_int3(v2, e20, e12);
    mesh.new_faces[4 * idx + 3] = make_int3(e01, e12, e20);
}

// gpu topology builders

__global__ void generate_edge_keys_kernel(const Face* faces, int num_faces, int2* out_keys, EdgeInfo* out_infos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_faces) return;

    Face f = faces[idx];
    int v[3] = {f.x, f.y, f.z};

    int i0 = v[0], j0 = v[1];
    out_keys[3*idx + 0] = (i0 < j0) ? make_int2(i0, j0) : make_int2(j0, i0);
    out_infos[3*idx + 0] = {idx, 0, v[2]};

    int i1 = v[1], j1 = v[2];
    out_keys[3*idx + 1] = (i1 < j1) ? make_int2(i1, j1) : make_int2(j1, i1);
    out_infos[3*idx + 1] = {idx, 1, v[0]};

    int i2 = v[2], j2 = v[0];
    out_keys[3*idx + 2] = (i2 < j2) ? make_int2(i2, j2) : make_int2(j2, i2);
    out_infos[3*idx + 2] = {idx, 2, v[1]};
}

__global__ void identify_unique_edges_kernel(
    const int2* sorted_keys,
    const EdgeInfo* sorted_infos,
    int num_raw_edges,
    int2* unique_edges,
    int2* edge_opposites,
    int3* face_edge_indices,
    int* edge_counter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_raw_edges) return;

    int2 my_key = sorted_keys[idx];
    bool is_start = (idx == 0) || (sorted_keys[idx-1].x != my_key.x || sorted_keys[idx-1].y != my_key.y);

    if (is_start) {
        int unique_id = atomicAdd(edge_counter, 1);
        unique_edges[unique_id] = my_key;

        bool has_twin = (idx + 1 < num_raw_edges) &&
                        (sorted_keys[idx+1].x == my_key.x && sorted_keys[idx+1].y == my_key.y);

        EdgeInfo info1 = sorted_infos[idx];
        int2 opposites = make_int2(info1.opposite_v, -1);

        int* face_map_ptr = (int*)face_edge_indices;
        face_map_ptr[3 * info1.face_idx + info1.edge_order] = unique_id;

        if (has_twin) {
            EdgeInfo info2 = sorted_infos[idx+1];
            opposites.y = info2.opposite_v;
            face_map_ptr[3 * info2.face_idx + info2.edge_order] = unique_id;
        }

        edge_opposites[unique_id] = opposites;
    }
}

__global__ void expand_adjacency_kernel(const int2* unique_edges, int num_unique_edges, int2* directed_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_unique_edges) return;

    int2 e = unique_edges[idx];
    directed_edges[2*idx]     = make_int2(e.x, e.y);
    directed_edges[2*idx + 1] = make_int2(e.y, e.x);
}

__global__ void copy_neighbors_kernel(const int2* directed_edges, int* adj_indices, int num_directed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_directed) {
        adj_indices[idx] = directed_edges[idx].y;
    }
}

__global__ void calc_valences_kernel(const int* offsets, int* valences, int num_vertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        valences[idx] = offsets[idx+1] - offsets[idx];
    }
}

// host stuff

void build_topology_gpu(DeviceMesh& d_mesh) {
    int num_raw_edges = d_mesh.num_faces * 3;

    thrust::device_vector<int2> d_raw_keys(num_raw_edges);
    thrust::device_vector<EdgeInfo> d_raw_infos(num_raw_edges);

    int threads = 256;
    int blocks = (num_raw_edges + threads - 1) / threads;
    generate_edge_keys_kernel<<<blocks, threads>>>(
        d_mesh.faces, d_mesh.num_faces,
        thrust::raw_pointer_cast(d_raw_keys.data()),
        thrust::raw_pointer_cast(d_raw_infos.data())
    );

    thrust::sort_by_key(d_raw_keys.begin(), d_raw_keys.end(), d_raw_infos.begin(), EdgeKeyCompare());

    cudaMalloc(&d_mesh.edge_indices, num_raw_edges * sizeof(int2));
    cudaMalloc(&d_mesh.edge_opposites, num_raw_edges * sizeof(int2));
    cudaMalloc(&d_mesh.odd_vertices, num_raw_edges * sizeof(Vertex));
    cudaMalloc(&d_mesh.face_edge_indices, d_mesh.num_faces * sizeof(int3));

    int* d_counter; cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    identify_unique_edges_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_raw_keys.data()),
        thrust::raw_pointer_cast(d_raw_infos.data()),
        num_raw_edges,
        d_mesh.edge_indices,
        d_mesh.edge_opposites,
        d_mesh.face_edge_indices,
        d_counter
    );

    int num_unique_edges;
    cudaMemcpy(&num_unique_edges, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    d_mesh.num_edges = num_unique_edges;
    cudaFree(d_counter);

    int num_directed = num_unique_edges * 2;
    thrust::device_vector<int2> d_directed(num_directed);

    expand_adjacency_kernel<<<(num_unique_edges + threads - 1) / threads, threads>>>(
        d_mesh.edge_indices, num_unique_edges, thrust::raw_pointer_cast(d_directed.data())
    );

    thrust::sort(d_directed.begin(), d_directed.end(), DirectedEdgeSort());

    cudaMalloc(&d_mesh.adj_offsets, (d_mesh.num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_mesh.adj_indices, num_directed * sizeof(int));
    cudaMalloc(&d_mesh.valences, d_mesh.num_vertices * sizeof(int));

    thrust::counting_iterator<int> search_begin(0);
    thrust::counting_iterator<int> search_end(d_mesh.num_vertices + 1);

    thrust::lower_bound(thrust::device,
                        d_directed.begin(), d_directed.end(),
                        search_begin, search_end,
                        thrust::device_pointer_cast(d_mesh.adj_offsets),
                        CompareInt2ToInt());

    copy_neighbors_kernel<<<(num_directed + threads - 1) / threads, threads>>>(
        thrust::raw_pointer_cast(d_directed.data()),
        d_mesh.adj_indices,
        num_directed
    );

    calc_valences_kernel<<<(d_mesh.num_vertices + threads - 1) / threads, threads>>>(
        d_mesh.adj_offsets, d_mesh.valences, d_mesh.num_vertices
    );
}

void cleanup_iteration(DeviceMesh& mesh) {
    if (mesh.adj_offsets) cudaFree(mesh.adj_offsets);
    if (mesh.adj_indices) cudaFree(mesh.adj_indices);
    if (mesh.valences) cudaFree(mesh.valences);
    if (mesh.edge_indices) cudaFree(mesh.edge_indices);
    if (mesh.edge_opposites) cudaFree(mesh.edge_opposites);
    if (mesh.odd_vertices) cudaFree(mesh.odd_vertices);
    if (mesh.face_edge_indices) cudaFree(mesh.face_edge_indices);

    mesh.adj_offsets = nullptr;
    mesh.adj_indices = nullptr;
    mesh.valences = nullptr;
    mesh.edge_indices = nullptr;
    mesh.edge_opposites = nullptr;
    mesh.odd_vertices = nullptr;
    mesh.face_edge_indices = nullptr;
}

// load, save, main

bool load_obj(const std::string& filename, HostMesh& mesh) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    mesh.vertices.clear(); mesh.faces.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string type; ss >> type;
        if (type == "v") {
            float x, y, z; ss >> x >> y >> z;
            mesh.vertices.push_back(make_float3(x, y, z));
        } else if (type == "f") {
            std::vector<int> f; std::string t;
            while (ss >> t) f.push_back(std::stoi(t.substr(0, t.find('/'))) - 1);
            if (f.size() == 3) mesh.faces.push_back(make_int3(f[0], f[1], f[2]));
            else if (f.size() == 4) {
                mesh.faces.push_back(make_int3(f[0], f[1], f[2]));
                mesh.faces.push_back(make_int3(f[0], f[2], f[3]));
            }
        }
    }
    return true;
}

bool save_obj_final(const std::string& filename, const std::vector<Vertex>& all_verts, const std::vector<Face>& all_faces) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << "# Generated by CUDA Loop Subdivision\n";
    for (const auto& v : all_verts) file << "v " << v.x << " " << v.y << " " << v.z << "\n";
    for (const auto& f : all_faces) file << "f " << f.x + 1 << " " << f.y + 1 << " " << f.z + 1 << "\n";
    return true;
}

__global__ void consolidate_vertices_kernel(
    Vertex* dst,
    const Vertex* even_src, int num_even,
    const Vertex* odd_src, int num_odd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_even) {
        dst[idx] = even_src[idx];
    } else if (idx < num_even + num_odd) {
        dst[idx] = odd_src[idx - num_even];
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./loop_cuda <input.obj> [iterations]\n";
        return 1;
    }
    int iterations = (argc > 2) ? std::stoi(argv[2]) : 1;

    HostMesh h_mesh;
    if (!load_obj(argv[1], h_mesh)) return 1;
    std::cout << "Loaded " << h_mesh.vertices.size() << " verts, " << h_mesh.faces.size() << " faces.\n";

    DeviceMesh d_mesh = {};
    d_mesh.num_vertices = h_mesh.vertices.size();
    d_mesh.num_faces = h_mesh.faces.size();

    cudaMalloc(&d_mesh.vertices, d_mesh.num_vertices * sizeof(Vertex));
    cudaMalloc(&d_mesh.faces, d_mesh.num_faces * sizeof(Face));
    cudaMemcpy(d_mesh.vertices, h_mesh.vertices.data(), d_mesh.num_vertices * sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.faces, h_mesh.faces.data(), d_mesh.num_faces * sizeof(Face), cudaMemcpyHostToDevice);

    // TIMING START
    using clock = std::chrono::high_resolution_clock;
    auto total_start = clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::cout << "\n--- Iteration " << (iter + 1) << " ---\n";

        auto iter_start = clock::now();

        // allocate output
        cudaMalloc(&d_mesh.new_vertices, d_mesh.num_vertices * sizeof(Vertex));
        cudaMalloc(&d_mesh.new_faces, (d_mesh.num_faces * 4) * sizeof(Face));

        // build topology (what we're timing as initialization)
        auto init_start = clock::now();

        build_topology_gpu(d_mesh);
        cudaDeviceSynchronize();

        auto init_end = clock::now();

        double init_seconds = std::chrono::duration<double>(init_end - init_start).count();
        std::cout << "Initialization time: " << std::fixed << std::setprecision(6) << init_seconds << " seconds\n";


        // run kernels
        int threads = 256;
        reposition_even_kernel<<<(d_mesh.num_vertices + threads - 1) / threads, threads>>>(d_mesh);
        compute_odd_vertices_kernel<<<(d_mesh.num_edges + threads - 1) / threads, threads>>>(d_mesh);
        rebuild_faces_kernel<<<(d_mesh.num_faces + threads - 1) / threads, threads>>>(d_mesh);
        cudaDeviceSynchronize();

        int next_num_vertices = d_mesh.num_vertices + d_mesh.num_edges;
        int next_num_faces = d_mesh.num_faces * 4;
        Vertex* d_next_vertices_buffer;
        cudaMalloc(&d_next_vertices_buffer, next_num_vertices * sizeof(Vertex));

        consolidate_vertices_kernel<<<(next_num_vertices + threads - 1) / threads, threads>>>(
            d_next_vertices_buffer,
            d_mesh.new_vertices, d_mesh.num_vertices,
            d_mesh.odd_vertices, d_mesh.num_edges
        );
        cudaDeviceSynchronize();

        cudaFree(d_mesh.vertices);
        cudaFree(d_mesh.faces);
        cudaFree(d_mesh.new_vertices);
        cleanup_iteration(d_mesh);

        d_mesh.vertices = d_next_vertices_buffer;
        d_mesh.faces = d_mesh.new_faces;
        d_mesh.num_vertices = next_num_vertices;
        d_mesh.num_faces = next_num_faces;
        d_mesh.new_vertices = nullptr;
        d_mesh.new_faces = nullptr;

        auto iter_end = clock::now();
        double iter_seconds = std::chrono::duration<double>(iter_end - iter_start).count();
        std::cout << "Iteration " << (iter + 1) << " time: " << std::fixed << std::setprecision(6) << iter_seconds << " seconds\n";
    }

    auto total_end = clock::now();
    double total_seconds = std::chrono::duration<double>(total_end - total_start).count();
    std::cout << "Total subdivision time (" << iterations << " iterations): "
              << std::fixed << std::setprecision(6)
              << total_seconds << " seconds\n";

    std::vector<Vertex> final_vertices(d_mesh.num_vertices);
    std::vector<Face> final_faces(d_mesh.num_faces);
    cudaMemcpy(final_vertices.data(), d_mesh.vertices, d_mesh.num_vertices * sizeof(Vertex), cudaMemcpyDeviceToHost);
    cudaMemcpy(final_faces.data(), d_mesh.faces, d_mesh.num_faces * sizeof(Face), cudaMemcpyDeviceToHost);
    save_obj_final("cuda_final_output.obj", final_vertices, final_faces);
    std::cout << "\nSaved 'cuda_final_output.obj'.\n";

    cudaFree(d_mesh.vertices);
    cudaFree(d_mesh.faces);
    return 0;
}

