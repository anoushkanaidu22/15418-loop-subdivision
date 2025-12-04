CXX = g++

CXXFLAGS = -O3 -std=c++17 -Wall

TARGET = loop_subdivision

SRCS = loop_subdivision.cpp

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.obj_subdiv_*.obj

.PHONY: all clean
