#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>
#include <algorithm>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

//struct for vertex positions
struct Vec3 {
    double x, y, z;
    Vec3() : x(0.0), y(0.0), z(0.0) {}
    Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    Vec3 operator*(double scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
    Vec3& operator=(const Vec3& other) {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }
};

//face struct that just stores the three vertices that make it up
//can't handle quads/other polygons
struct Face {
    int v[3];
};

//using an adjacency list
struct Mesh {
    std::vector<Vec3> V; //vertex buffer
    std::vector<Face> F; //face buffer
    std::vector<std::vector<int>> neighbors;
    std::map<std::pair<int, int>, std::vector<int>> edge_faces;
    int num_vertices = 0;
    int num_faces = 0;
};

//fn so that edge is same no matter which order you reference it (aka (x,y) or (y,x))
std::pair<int, int> make_edge_key(int i, int j) {
    return (i < j) ? std::make_pair(i, j) : std::make_pair(j, i);
}

void loop_subdivide_sequential(Mesh& mesh) {
    //building adjacency
    mesh.neighbors.assign(mesh.V.size(), std::vector<int>());
    mesh.edge_faces.clear();
    std::vector<std::map<int, bool>> unique_neighbors(mesh.V.size());

    for (int f_idx = 0; f_idx < mesh.F.size(); ++f_idx) {
        const Face& f = mesh.F[f_idx];
        int v_i = f.v[0];
        int v_j = f.v[1];
        int v_k = f.v[2];

        //prevent duplicate neighbors from  edges shared between faces
        unique_neighbors[v_i][v_j] = true;
        unique_neighbors[v_i][v_k] = true;
        unique_neighbors[v_j][v_i] = true;
        unique_neighbors[v_j][v_k] = true;
        unique_neighbors[v_k][v_i] = true;
        unique_neighbors[v_k][v_j] = true;
        std::pair<int, int> edges[3] = {
            make_edge_key(v_i, v_j),
            make_edge_key(v_j, v_k),
            make_edge_key(v_k, v_i)
        };
        for (int i = 0; i < 3; ++i) {
            mesh.edge_faces[edges[i]].push_back(f_idx);
        }
    }
    for (int i = 0; i < mesh.V.size(); ++i) {
        for (const auto& pair : unique_neighbors[i]) {
            mesh.neighbors[i].push_back(pair.first);
        }
    }

    //phase 1: new positions for even/old verticies
    std::vector<Vec3> V_even(mesh.V.size());
    for (int i = 0; i < mesh.V.size(); ++i) {
        const std::vector<int>& N = mesh.neighbors[i];
        int n = N.size();
        Vec3 sumN(0, 0, 0);
        double beta = 0.0;
        double alpha = 1.0;
        bool is_boundary_v = false;
        std::vector<int> boundary_neighbors;
        for (int neighbor_idx : N) {
            std::pair<int, int> edge = make_edge_key(i, neighbor_idx);
            if (mesh.edge_faces.count(edge) && mesh.edge_faces.at(edge).size() == 1) {
                is_boundary_v = true;
                boundary_neighbors.push_back(neighbor_idx);
            }
        }
        if (is_boundary_v) {
            if (boundary_neighbors.size() == 2) {
                sumN = mesh.V[boundary_neighbors[0]] + mesh.V[boundary_neighbors[1]];
                beta = 1.0 / 8.0;
                alpha = 6.0 / 8.0;
            } else {
                if (n == 0) { V_even[i] = mesh.V[i]; continue; }
                for (int neighbor_idx : N) sumN = sumN + mesh.V[neighbor_idx];
                if (n == 3) beta = 3.0 / 16.0;
                else beta = (1.0 / (double)n) * (5.0 / 8.0 - pow(3.0 / 8.0 + 0.25 * cos(2.0 * M_PI / n), 2));
                alpha = 1.0 - (double)n * beta;
            }
        } else {
            if (n == 0) { V_even[i] = mesh.V[i]; continue; }
            for (int neighbor_idx : N) sumN = sumN + mesh.V[neighbor_idx];
            if (n == 3) beta = 3.0 / 16.0;
            else beta = (1.0 / (double)n) * (5.0 / 8.0 - pow(3.0 / 8.0 + 0.25 * cos(2.0 * M_PI / n), 2));
            alpha = 1.0 - (double)n * beta;
        }
        V_even[i] = mesh.V[i] * alpha + sumN * beta;
    }

    //phase 2: creating odd/new vertices (one for each og edge)
    std::vector<Vec3> V_new;
    std::map<std::pair<int, int>, int> edge_to_new_v_idx;
    V_new.insert(V_new.end(), V_even.begin(), V_even.end());
    for (const auto& pair : mesh.edge_faces) {
        std::pair<int, int> edge = pair.first;
        const std::vector<int>& faces = pair.second;
        int v_i = edge.first;
        int v_j = edge.second;
        Vec3 V_odd;
        if (faces.size() == 1) { // boundary
            V_odd = (mesh.V[v_i] + mesh.V[v_j]) * 0.5;
        } else if (faces.size() == 2) { //interior
            int v_opp_L = -1, v_opp_R = -1;
            const Face& f_L = mesh.F[faces[0]];
            for (int k = 0; k < 3; ++k) if (f_L.v[k] != v_i && f_L.v[k] != v_j) v_opp_L = f_L.v[k];
            const Face& f_R = mesh.F[faces[1]];
            for (int k = 0; k < 3; ++k) if (f_R.v[k] != v_i && f_R.v[k] != v_j) v_opp_R = f_R.v[k];
            if (v_opp_L != -1 && v_opp_R != -1) {
                V_odd = (mesh.V[v_i] + mesh.V[v_j]) * (3.0/8.0) + (mesh.V[v_opp_L] + mesh.V[v_opp_R]) * (1.0/8.0);
            } else {
                V_odd = (mesh.V[v_i] + mesh.V[v_j]) * 0.5;
            }
        } else {
            V_odd = (mesh.V[v_i] + mesh.V[v_j]) * 0.5;
        }
        int new_v_idx = V_new.size();
        V_new.push_back(V_odd);
        edge_to_new_v_idx[edge] = new_v_idx;
    }

    //phase 3: rebuilding faces
    std::vector<Face> F_new;
    for (const Face& old_face : mesh.F) {
        int v_i = old_face.v[0];
        int v_j = old_face.v[1];
        int v_k = old_face.v[2];
        int e_ij = edge_to_new_v_idx.at(make_edge_key(v_i, v_j));
        int e_jk = edge_to_new_v_idx.at(make_edge_key(v_j, v_k));
        int e_ki = edge_to_new_v_idx.at(make_edge_key(v_k, v_i));
        F_new.push_back({{v_i, e_ij, e_ki}});
        F_new.push_back({{v_j, e_jk, e_ij}});
        F_new.push_back({{v_k, e_ki, e_jk}});
        F_new.push_back({{e_ij, e_jk, e_ki}});
    }

    //phase 4: update mesh
    mesh.V = std::move(V_new);
    mesh.F = std::move(F_new);
    mesh.num_vertices = mesh.V.size();
    mesh.num_faces = mesh.F.size();
}

// everything below is for parsing input obj file and writing output obj file

bool load_obj(const std::string& filename, Mesh& mesh) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    mesh.V.clear();
    mesh.F.clear();

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string type;
        ss >> type;

        if (type == "v") {
            double x, y, z;
            if (ss >> x >> y >> z) mesh.V.emplace_back(x, y, z);
        } else if (type == "f") {
            // Read all vertices in the face line to handle Quads
            std::vector<int> face_indices;
            std::string token;

            while (ss >> token) {
                size_t pos = token.find('/');
                std::string v_str = (pos == std::string::npos) ? token : token.substr(0, pos);
                try {
                    face_indices.push_back(std::stoi(v_str) - 1);
                } catch (...) {
                    continue;
                }
            }

            //added this so we can handle quad meshes (just triangulates them) but didn't add an overall triangulation fn so this won't work for an n-gon for n > 4
            if (face_indices.size() == 3) {
                mesh.F.push_back({{face_indices[0], face_indices[1], face_indices[2]}});
            } else if (face_indices.size() == 4) {
                mesh.F.push_back({{face_indices[0], face_indices[1], face_indices[2]}});
                mesh.F.push_back({{face_indices[0], face_indices[2], face_indices[3]}});
            } else if (face_indices.size() > 4) {
                std::cerr << "polygon with " << face_indices.size() << " which we can't handle" << std::endl;
            }
        }
    }

    mesh.num_vertices = mesh.V.size();
    mesh.num_faces = mesh.F.size();
    std::cout << "Loaded mesh " << filename << ": " << mesh.num_vertices << " vertices, " << mesh.num_faces << " faces." << std::endl;
    return true;
}

bool save_obj(const std::string& filename, const Mesh& mesh) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << filename << std::endl;
        return false;
    }

    file << std::fixed << std::setprecision(8);
    file << "# Generated by Sequential Loop Subdivision" << std::endl;
    file << "# Vertices: " << mesh.V.size() << ", Faces: " << mesh.F.size() << std::endl;

    for (const auto& v : mesh.V) {
        file << "v " << v.x << " " << v.y << " " << v.z << std::endl;
    }

    for (const auto& f : mesh.F) {
        file << "f " << f.v[0] + 1 << " " << f.v[1] + 1 << " " << f.v[2] + 1 << std::endl;
    }

    std::cout << "Saved resulting mesh to " << filename << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_obj_filename> <num_iterations>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1];
    int num_iters;
    try {
        num_iters = std::stoi(argv[2]);
    } catch (...) {
        return 1;
    }

    Mesh mesh;
    if (!load_obj(input_filename, mesh)) return 1;

    for (int i = 1; i <= num_iters; ++i) {
        std::cout << "--- Iteration " << i << "/" << num_iters << " ---" << std::endl;
        loop_subdivide_sequential(mesh);
    }

    size_t dot_pos = input_filename.find_last_of('.');
    std::string output_filename = (dot_pos == std::string::npos)
        ? input_filename + "_subdiv_" + std::to_string(num_iters) + ".obj"
        : input_filename.substr(0, dot_pos) + "_subdiv_" + std::to_string(num_iters) + ".obj";

    save_obj(output_filename, mesh);
    return 0;
}
