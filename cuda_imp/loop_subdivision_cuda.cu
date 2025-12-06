#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cuda_runtime.h>

#include "mesh.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

std::pair<int, int> make_edge_key(int i, int j) {
    return (i < j) ? std::make_pair(i, j) : std::make_pair(j, i);
}

// kernel for repositioning even vertices
__global__ void reposition_even_kernel(DeviceMesh mesh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= mesh.num_vertices) return;
    Vertex v_curr = mesh.vertices[idx];
    int n = mesh.valences[idx];
    int start_idx = mesh.adj_offsets[idx];
    Vertex sum_neighbors = make_float3(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < n; ++k) {
        int neighbor_idx = mesh.adj_indices[start_idx + k];
        sum_neighbors = sum_neighbors + mesh.vertices[neighbor_idx];
    }
    float beta;
    if (n == 3) {
        beta = 3.0f / 16.0f;
    } else {
        float term = 3.0f / 8.0f + 0.25f * cosf(2.0f * M_PI / n);
        beta = (1.0f / n) * (5.0f / 8.0f - term * term);
    }
    float alpha = 1.0f - (n * beta);
    mesh.new_vertices[idx] = (v_curr * alpha) + (sum_neighbors * beta);
}

// kernel for creating odd verticies
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

//kerenl for rebuilding faces
__global__ void rebuild_faces_kernel(DeviceMesh mesh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= mesh.num_faces) return;
    Face old_face = mesh.faces[idx];
    int v0 = old_face.x;
    int v1 = old_face.y;
    int v2 = old_face.z;
    int3 edge_idxs = mesh.face_edge_indices[idx];
    int e01 = mesh.num_vertices + edge_idxs.x; // edge(v0, v1)
    int e12 = mesh.num_vertices + edge_idxs.y; // edge(v1, v2)
    int e20 = mesh.num_vertices + edge_idxs.z; // edge(v2, v0)

    // need to create 4 new faces
    // face 1: top (v0, e01, e20)
    mesh.new_faces[4 * idx + 0] = make_int3(v0, e01, e20);

    // face 2: right (v1, e12, e01)
    mesh.new_faces[4 * idx + 1] = make_int3(v1, e12, e01);

    // face 3: left (v2, e20, e12)
    mesh.new_faces[4 * idx + 2] = make_int3(v2, e20, e12);

    // face 4: center (e01, e12, e20)
    mesh.new_faces[4 * idx + 3] = make_int3(e01, e12, e20);
}

// host toplogy building, will need a lot of change to do on gpu but we have to do that
// same as sequential basically

void build_topology_host(HostMesh& mesh) {
    mesh.adjacency_list.assign(mesh.vertices.size(), std::vector<int>());
    std::vector<std::map<int, bool>> unique_neighbors(mesh.vertices.size());
    std::map<std::pair<int, int>, std::vector<int>> edge_to_opposites;
    std::map<std::pair<int, int>, int> edge_to_index_map;
    int edge_counter = 0;
    for (const auto& f : mesh.faces) {
        int v[3] = { f.x, f.y, f.z };

        unique_neighbors[v[0]][v[1]] = true; unique_neighbors[v[0]][v[2]] = true;
        unique_neighbors[v[1]][v[0]] = true; unique_neighbors[v[1]][v[2]] = true;
        unique_neighbors[v[2]][v[0]] = true; unique_neighbors[v[2]][v[1]] = true;

        std::pair<int, int> edges[3] = {
            make_edge_key(v[0], v[1]),
            make_edge_key(v[1], v[2]),
            make_edge_key(v[2], v[0])
        };

        for (int i = 0; i < 3; ++i) {
            if (edge_to_index_map.find(edges[i]) == edge_to_index_map.end()) {
                edge_to_index_map[edges[i]] = edge_counter++;
            }
        }
        edge_to_opposites[edges[0]].push_back(v[2]);
        edge_to_opposites[edges[1]].push_back(v[0]);
        edge_to_opposites[edges[2]].push_back(v[1]);
    }

    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        for (auto const& [neighbor, _] : unique_neighbors[i]) {
            mesh.adjacency_list[i].push_back(neighbor);
        }
    }

    mesh.unique_edges.resize(edge_counter);
    mesh.edge_opposites.resize(edge_counter);
    for (auto const& [edge_key, idx] : edge_to_index_map) {
        mesh.unique_edges[idx] = make_int2(edge_key.first, edge_key.second);
        const auto& opposites = edge_to_opposites[edge_key];
        int2 opp_indices = make_int2(-1, -1);
        if (opposites.size() >= 1) opp_indices.x = opposites[0];
        if (opposites.size() >= 2) opp_indices.y = opposites[1];
        mesh.edge_opposites[idx] = opp_indices;
    }

    mesh.face_edge_indices.clear();
    for (const auto& f : mesh.faces) {
        int v0 = f.x; int v1 = f.y; int v2 = f.z;
        int e01 = edge_to_index_map[make_edge_key(v0, v1)];
        int e12 = edge_to_index_map[make_edge_key(v1, v2)];
        int e20 = edge_to_index_map[make_edge_key(v2, v0)];
        mesh.face_edge_indices.push_back(make_int3(e01, e12, e20));
    }
}

// obj parser
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

int main(int argc, char* argv[]) {
    if (argc != 2) { std::cout << "Usage: ./loop_cuda <input.obj>\n"; return 1; }
    HostMesh h_mesh;
    if (!load_obj(argv[1], h_mesh)) return 1;
    build_topology_host(h_mesh);
    std::cout << "loaded: " << h_mesh.vertices.size() << " verts, " << h_mesh.faces.size() << " faces, " << h_mesh.unique_edges.size() << " edges.\n";
    std::vector<int> h_adj_offsets, h_adj_indices, h_valences;
    int current_offset = 0;
    for (const auto& neighbors : h_mesh.adjacency_list) {
        h_adj_offsets.push_back(current_offset);
        h_valences.push_back(neighbors.size());
        for (int n : neighbors) h_adj_indices.push_back(n);
        current_offset += neighbors.size();
    }
    DeviceMesh d_mesh;
    d_mesh.num_vertices = h_mesh.vertices.size();
    d_mesh.num_faces = h_mesh.faces.size();
    d_mesh.num_edges = h_mesh.unique_edges.size();
    cudaMalloc(&d_mesh.vertices, d_mesh.num_vertices * sizeof(Vertex));
    cudaMalloc(&d_mesh.faces, d_mesh.num_faces * sizeof(Face));
    cudaMalloc(&d_mesh.new_vertices, d_mesh.num_vertices * sizeof(Vertex));
    cudaMalloc(&d_mesh.adj_offsets, h_adj_offsets.size() * sizeof(int));
    cudaMalloc(&d_mesh.adj_indices, h_adj_indices.size() * sizeof(int));
    cudaMalloc(&d_mesh.valences, h_valences.size() * sizeof(int));

    cudaMalloc(&d_mesh.odd_vertices, d_mesh.num_edges * sizeof(Vertex));
    cudaMalloc(&d_mesh.edge_indices, d_mesh.num_edges * sizeof(int2));
    cudaMalloc(&d_mesh.edge_opposites, d_mesh.num_edges * sizeof(int2));

    cudaMalloc(&d_mesh.face_edge_indices, d_mesh.num_faces * sizeof(int3));
    cudaMalloc(&d_mesh.new_faces, (d_mesh.num_faces * 4) * sizeof(Face));

    cudaMemcpy(d_mesh.vertices, h_mesh.vertices.data(), d_mesh.num_vertices * sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.faces, h_mesh.faces.data(), d_mesh.num_faces * sizeof(Face), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mesh.adj_offsets, h_adj_offsets.data(), h_adj_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.adj_indices, h_adj_indices.data(), h_adj_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.valences, h_valences.data(), h_valences.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mesh.edge_indices, h_mesh.unique_edges.data(), d_mesh.num_edges * sizeof(int2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mesh.edge_opposites, h_mesh.edge_opposites.data(), d_mesh.num_edges * sizeof(int2), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mesh.face_edge_indices, h_mesh.face_edge_indices.data(), d_mesh.num_faces * sizeof(int3), cudaMemcpyHostToDevice);

    int threads = 256;

    reposition_even_kernel<<<(d_mesh.num_vertices + threads - 1) / threads, threads>>>(d_mesh);
    compute_odd_vertices_kernel<<<(d_mesh.num_edges + threads - 1) / threads, threads>>>(d_mesh);
    rebuild_faces_kernel<<<(d_mesh.num_faces + threads - 1) / threads, threads>>>(d_mesh);

    cudaDeviceSynchronize();

    int total_vertices = d_mesh.num_vertices + d_mesh.num_edges;
    int total_faces = d_mesh.num_faces * 4;

    std::vector<Vertex> final_vertices(total_vertices);
    std::vector<Face> final_faces(total_faces);

    cudaMemcpy(final_vertices.data(), d_mesh.new_vertices, d_mesh.num_vertices * sizeof(Vertex), cudaMemcpyDeviceToHost);
    cudaMemcpy(final_vertices.data() + d_mesh.num_vertices, d_mesh.odd_vertices, d_mesh.num_edges * sizeof(Vertex), cudaMemcpyDeviceToHost);
    cudaMemcpy(final_faces.data(), d_mesh.new_faces, total_faces * sizeof(Face), cudaMemcpyDeviceToHost);
    save_obj_final("cuda_final_output.obj", final_vertices, final_faces);
    std::cout << "subdivision done and saved 'cuda_final_output.obj' (" << total_vertices << " vertices, " << total_faces << " faces).\n";
    cudaFree(d_mesh.vertices); cudaFree(d_mesh.new_vertices); cudaFree(d_mesh.faces);
    cudaFree(d_mesh.adj_offsets); cudaFree(d_mesh.adj_indices); cudaFree(d_mesh.valences);
    cudaFree(d_mesh.odd_vertices); cudaFree(d_mesh.edge_indices); cudaFree(d_mesh.edge_opposites);
    cudaFree(d_mesh.face_edge_indices); cudaFree(d_mesh.new_faces);

    return 0;
}
