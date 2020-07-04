TF_INC ?= `python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`
TF_LIB ?= `python -c "import tensorflow; print(tensorflow.sysconfig.get_lib())"`
CUDA_HOME ?= /usr/local/cuda

SRC_DIR = ./lib/ops

BUILD_DIR = ./lib/ops/build
LIB_DIR = ./lib/ops/lib

CC = c++ -std=c++11
NVCC = nvcc -std c++11
CFLAGS = -fPIC -I$(TF_INC) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -DGOOGLE_CUDA=1
LDFLAGS = -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -L$(TF_LIB) -ltensorflow_framework
NVFLAGS = -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I$(TF_INC) -c -I /usr/local\
					-expt-relaxed-constexpr -Wno-deprecated-gpu-targets -ftz=true\
					-O2 -D_GLIBCXX_USE_CXX11_ABI=0

SRC = deform_conv.cc
CUDA_SRC = deform_conv.cu.cc
CUDA_OBJ = $(addprefix $(BUILD_DIR)/,$(CUDA_SRC:.cc=.o))
SRCS = $(addprefix $(SRC_DIR)/, $(SRC))

all: $(LIB_DIR)/deform_conv.so

# Main library
$(LIB_DIR)/deform_conv.so: $(CUDA_OBJ) $(LIB_DIR) $(SRCS)
	$(CC) -shared -o $@ $(SRCS) $(CUDA_OBJ) $(CFLAGS) $(LDFLAGS)

# Cuda kernels
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc $(BUILD_DIR)
	$(NVCC) -c  $< -o $@ $(NVFLAGS)

$(BUILD_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)
