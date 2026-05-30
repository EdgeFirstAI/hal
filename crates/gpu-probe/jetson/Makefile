# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# On-device build for the Jetson CUDA/TensorRT NvBufSurface probe (O4-O7).
# Build on orin-nano (L4T R36.4, CUDA 12.6, TensorRT 10.3):
#     make
#     ./jetson_cuda_probe
#
# This is on-device tooling — it is NOT cross-compiled from the x86 host
# (no aarch64 NVIDIA sysroot). nvcc compiles the CUDA/EGL/TRT interop.

NVCC      ?= /usr/local/cuda-12.6/bin/nvcc

# NvBufSurface + CUDA + TensorRT headers. NvInfer.h ships under the
# arch triplet include dir on this image.
INCLUDES  := -I/usr/src/jetson_multimedia_api/include \
             -I/usr/local/cuda/include \
             -I/usr/include/aarch64-linux-gnu

# nvbufsurface lives under the nvidia subdir; the rest are standard.
LIBDIRS   := -L/usr/lib/aarch64-linux-gnu/nvidia \
             -L/usr/lib/aarch64-linux-gnu
LIBS      := -lnvbufsurface -lcudart -lcuda -lnvinfer -lEGL -lGLESv2

TARGET    := jetson_cuda_probe
SRC       := jetson_cuda_probe.cpp

$(TARGET): $(SRC)
	$(NVCC) -O2 -std=c++17 $(INCLUDES) $(SRC) -o $(TARGET) $(LIBDIRS) $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
