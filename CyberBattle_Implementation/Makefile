#########################
# Project configuration #
#########################

# Compilers
CXX  := g++
NVCC := /home1/apps/cuda/12.2/bin/nvcc

# Paths (adjust if your env paths ever change)
CUDA_HOME   := /home1/apps/cuda/12.2
PY38_ENV    := /home1/11155/victoruche11/miniconda3/envs/py38
GCC_LIB_DIR := /opt/apps/gcc/12.2.0/lib64

# Include paths
INC_CUDA      := -I$(CUDA_HOME)/include
INC_PYTHON    := -I$(PY38_ENV)/include/python3.8
INC_PYBIND11  := -I$(PY38_ENV)/lib/python3.8/site-packages/pybind11/include

# Common compile flags
CXXFLAGS := -std=c++14 -O3 $(INC_CUDA) $(INC_PYTHON) $(INC_PYBIND11)
NVCCFLAGS := -std=c++14 -O3 -arch=sm_70 $(INC_CUDA) $(INC_PYTHON) $(INC_PYBIND11)

# Libraries and linker flags
LDFLAGS := \
    -L$(CUDA_HOME)/lib64 -lcudart -lcurand \
    -L$(GCC_LIB_DIR) -lstdc++ \
    -L$(PY38_ENV)/lib -lpython3.8 \
    -pthread \
    -Wl,-rpath,$(CUDA_HOME)/lib64 \
    -Wl,-rpath,$(GCC_LIB_DIR) \
    -Wl,-rpath,$(PY38_ENV)/lib

# Object files for main executable
OBJS := \
    main.o \
    CudaNetworkEnv.o \
    CyberBattleEnv.o \
    HybridEnv.o \
    TabularQLearning.o \
    VulnerabilityAnalyzer.o \
    cuda_kernels.o

# Output binary
TARGET := cyber_rl


#################
# Build targets #
#################

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

# Compile C++ sources with g++
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source with nvcc
cuda_kernels.o: cuda_kernels.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


############
# Cleaning #
############

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean