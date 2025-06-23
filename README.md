# EdgeFirst Hardware Abstraction Layer

The EdgeFirst Hardware Abstraction Layer "HAL" provides a set of generic APIs for common tasks required to support the EdgeFirst Perception Middleware and user applications.  The HAL abstracts inference engines, DMA or Shared Memory buffer creation and sharing, fast image decoding and resizing.  The HAL provides generic CPU fallback for operations but is optimized for leveraging device-specific hardware accelerators for these tasks.

## Model HAL

The Model HAL abstracts model loading, buffer management (using the Buffer HAL), inferencing, and querying of embedded model meta-data.  It supports various engines under-the-hood such as ONNX, TFLite/LiteRT, and proprietary engines such as the Kinara Ara-2.

## Tensor HAL

The Tensor HAL is optimized for Linux DMA Heap allocations and sharing with fallback for portable Shared Memory implementations supporting Linux as well as MacOS and Windows.

## Image HAL

The Image HAL abstracts image conversion and resize operations using hardware accelerators such as G2D, PXP, and VPI.
