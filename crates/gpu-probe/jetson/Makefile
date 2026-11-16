# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# On-device build for the Jetson CUDA/TensorRT NvBufSurface probe:
#     make
#     ./jetson_cuda_probe

NVCC      ?= /usr/local/cuda-12.6/bin/nvcc

# NvBufSurface, CUDA and TensorRT headers.
INCLUDES  := -I/usr/src/jetson_multimedia_api/include \
             -I/usr/local/cuda/include \
             -I/usr/include/aarch64-linux-gnu

# nvbufsurface is under the nvidia subdirectory.
LIBDIRS   := -L/usr/lib/aarch64-linux-gnu/nvidia \
             -L/usr/lib/aarch64-linux-gnu
LIBS      := -lnvbufsurface -lcudart -lcuda -lnvinfer -lEGL -lGLESv2 -lgbm

TARGET    := jetson_cuda_probe
SRC       := jetson_cuda_probe.cpp

$(TARGET): $(SRC)
	$(NVCC) -O2 -std=c++20 $(INCLUDES) $(SRC) -o $(TARGET) $(LIBDIRS) $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
