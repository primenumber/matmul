CXX=g++-6
CU=nvcc
CXXFLAGS=-std=c++1z -march=native -mtune=native -Og -g -Wall -Wextra -fPIC
CUFLAGS=-std=c++11 -arch=sm_61
LDFLAGS=-lboost_system -lboost_timer -lpthread -L/opt/cuda/lib64 -lcuda -lcudart
.SUFFIXES: .cpp .c .cu .o

matmul: matmul_cpu.o matmul_gpu.o matmul.o
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^

matmul_cpu_g: matmul_cpu.cpp matmul_common.hpp
	$(CXX) -o $@ -g $(CXXFLAGS) $(LDFLAGS) $<

matmul_cpu.s: matmul_cpu.cpp matmul_cpu.hpp
	$(CXX) -o $@ -S $(CXXFLAGS) $(LDFLAGS) $<

matmul_gpu.cubin: matmul_gpu.cu matmul_gpu.h
	$(CU) -o $@ -cubin $(CUFLAGS) $<

matmul_gpu.sass: matmul_gpu.cubin
	cuobjdump -sass $< > $@

.cu.o:
	$(CU) -c $(CUFLAGS) $<

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $<

matmul.o: matmul_common.hpp matmul_cpu.hpp matmul_gpu.h
matmul_gpu.o: matmul_gpu.h
matmul_cpu.o: matmul_cpu.hpp
