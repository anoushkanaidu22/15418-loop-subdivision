NVCC = nvcc
CXX = g++-11

NVCCFLAGS = -O3 -m64 -arch=sm_75 -ccbin /usr/bin/g++-11

LDFLAGS = -L/usr/local/cuda-11.7/lib64/ -lcudart

TARGET = loop_cuda
SRCS = loop_subdivision_cuda.cu
HEADERS = mesh.h

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.obj_subdiv_*.obj

.PHONY: all clean
