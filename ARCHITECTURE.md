# EdgeFirst Hardware Abstraction Layer - Architecture

**Version:** 2.9
**Last Updated:** March 23, 2026
**Status:** Production
**Audience:** Developers contributing to EdgeFirst HAL or integrating it into applications

---

## Overview

The EdgeFirst Hardware Abstraction Layer (HAL) is a Rust-based system that provides hardware-accelerated abstractions for computer vision and machine learning tasks on embedded Linux platforms. The HAL consists of multiple specialized crates that work together to provide high-performance image processing, tensor operations, model inference decoding, and object tracking.

## System Architecture

```mermaid
graph TB
    subgraph "EdgeFirst HAL"
        Python["Python Bindings (edgefirst-hal)<br/>PyO3-based Python API"]
        CAPI["C API Bindings (edgefirst-hal-capi)<br/>cbindgen-generated C headers"]
        Main["Main HAL Crate (edgefirst)<br/>Re-exports tensor, image, decoder"]

        Python --> Main
        CAPI --> Main

        Tensor["Tensor HAL<br/>Zero-copy memory buffers"]
        Image["Image Converter HAL<br/>Format conversion & resize"]
        Decoder["Decoder HAL<br/>Model output post-processing"]
        Tracker["Tracker HAL<br/>Multi-object tracking"]

        Main --> Tensor
        Main --> Image
        Main --> Decoder

        Image --> Tensor
        Image --> Decoder
        Image --> G2D["G2D FFI (g2d-sys)<br/>NXP i.MX hardware acceleration"]
        CAPI --> Tracker
    end

    Tensor -.-> DMA["Linux DMA-Heap<br/>Shared Memory"]
    Decoder -.-> PostProc["Model Output<br/>Post-Processing"]

    style Python fill:#e1f5ff
    style CAPI fill:#e1f5ff
    style Main fill:#fff4e1
    style Tensor fill:#e8f5e9
    style Image fill:#e8f5e9
    style Decoder fill:#e8f5e9
    style Tracker fill:#e8f5e9
```

## Core Components

### 1. Tensor HAL (`edgefirst_tensor`)

**Purpose**: Provides zero-copy memory buffers optimized for hardware accelerators.

**Architecture**:
```mermaid
classDiagram
    class TensorTrait~T~ {
        <<trait>>
        +shape() Vec~usize~
        +size() usize
        +map() TensorMap~T~
        +clone_fd() Result~i32~
    }
    
    class DmaTensor~T~ {
        Linux DMA-Heap allocation
    }
    
    class ShmTensor~T~ {
        POSIX Shared Memory
    }
    
    class MemTensor~T~ {
        Standard heap allocation
    }

    class PboTensor~T~ {
        OpenGL Pixel Buffer Object
    }

    TensorTrait <|.. DmaTensor
    TensorTrait <|.. ShmTensor
    TensorTrait <|.. MemTensor
    TensorTrait <|.. PboTensor
```

**Key Features**:
- Generic over numeric types (`T: Num + Clone + Debug + Send + Sync`) — commonly u8, i8, u16, i16, u32, i32, f32, f64
- Automatic memory type selection with fallback chain: DMA → Shared Memory → Heap
- PBO (Pixel Buffer Object) tensors for GPU-accelerated image processing
- Memory mapping with `TensorMap<T>` for safe access (RAII map/unmap lifecycle)
- `BufferIdentity` for cache keying and liveness tracking (monotonic ID + `Arc<()>` guard with weak reference detection)
- File descriptor sharing for zero-copy IPC (DMA-buf and SHM both support fd cloning)
- Cross-platform support (Linux optimized, macOS via SHM + heap, Windows via heap only)

**Tensor Memory Mapping**:

Each tensor backend provides a corresponding map type that implements `TensorMapTrait<T>`:

| Tensor | Map | Mechanism |
|--------|-----|-----------|
| `DmaTensor<T>` | `DmaMap<T>` | `mmap` + `DMA_BUF_IOCTL_SYNC` for cache coherency |
| `ShmTensor<T>` | `ShmMap<T>` | `mmap`/`munmap` on POSIX shared memory fd |
| `MemTensor<T>` | `MemMap<T>` | Direct raw pointer into `Vec<T>` (no syscall) |
| `PboTensor<T>` | `PboMap<T>` | GL thread `glMapBufferRange`/`glUnmapBuffer` via channel |

`TensorMap<T>` implements `Deref<Target=[T]>` and `DerefMut`, providing slice access. When the `ndarray` feature is enabled, `TensorMapTrait` also provides `view()` and `view_mut()` for ndarray `ArrayView` access.

**Memory Type Selection Logic**:
```mermaid
flowchart TD
    Start[User Request] --> Explicit{Explicit type?}
    Explicit -->|Yes| UseSpec[Use specified type]
    Explicit -->|No| CheckEnv{EDGEFIRST_TENSOR_FORCE_MEM=1?}
    CheckEnv -->|Yes| UseMem[MemTensor]
    CheckEnv -->|No| TryDMA[Try DmaTensor]
    TryDMA --> DMASuccess{Success?}
    DMASuccess -->|Yes| UseDMA[DmaTensor]
    DMASuccess -->|No| TryShm[Try ShmTensor]
    TryShm --> ShmSuccess{Success?}
    ShmSuccess -->|Yes| UseShm[ShmTensor]
    ShmSuccess -->|No| UseMem
    
    style UseDMA fill:#90ee90
    style UseShm fill:#87ceeb
    style UseMem fill:#ffeb9c
```

**PBO Tensor Memory (`PboTensor<T>`)**:

PBO tensors are a GPU-native memory type created by the OpenGL backend when
`ImageProcessor::create_image()` is called and DMA-buf is not available. Unlike
the other tensor types, PBO tensors are not allocated by the tensor crate
directly — they are OpenGL Pixel Buffer Objects managed by the GL thread. The
tensor crate provides the `PboTensor` wrapper and the `PboOps` trait that the
GL backend implements to perform map/unmap/delete operations.

PBO tensors use a `WeakSender` to communicate with the GL thread. This is a
critical design choice: if `PboTensor` held a strong `Sender`, any surviving
PBO tensor would keep the GL thread's message channel alive, preventing
`GLProcessorThreaded::drop()` from joining the GL thread at shutdown. The
`WeakSender` pattern allows the GL thread to exit cleanly when the
`ImageProcessor` is dropped, even if PBO tensors still exist — subsequent
PBO operations on orphaned tensors return `PboDisconnected`.

### 2. Image HAL (`edgefirst_image`)

**Purpose**: Hardware-accelerated image format conversion and resizing.

**Architecture**:
```mermaid
classDiagram
    class ImageProcessorTrait {
        <<trait>>
        +convert(src, dst, rotation, flip, crop)
        +draw_masks(dst, detections, segmentations)
        +draw_masks_proto(dst, detections, proto_data)
        +decode_masks_atlas(detections, proto_data, w, h)
        +set_class_colors(colors)
    }

    class ImageProcessor {
        cpu: Option~CPUProcessor~
        g2d: Option~G2DProcessor~
        opengl: Option~GLProcessorThreaded~
        +new() orchestrator with fallback chain
        +create_image(w, h, PixelFormat, DType, Option~TensorMemory~) GPU-optimal alloc
    }

    class G2DProcessor {
        NXP i.MX G2D hardware
    }

    class GLProcessorThreaded {
        Threaded OpenGL ES wrapper
        sends messages to GL thread
    }

    class GLProcessorST {
        Single-threaded GL impl
        owns EGL context + all GL state
    }

    class CPUProcessor {
        fast_image_resize + rayon
    }

    ImageProcessorTrait <|.. ImageProcessor
    ImageProcessorTrait <|.. G2DProcessor
    ImageProcessorTrait <|.. GLProcessorThreaded
    ImageProcessorTrait <|.. GLProcessorST
    ImageProcessorTrait <|.. CPUProcessor
    ImageProcessor o-- G2DProcessor
    ImageProcessor o-- GLProcessorThreaded
    ImageProcessor o-- CPUProcessor
    GLProcessorThreaded *-- GLProcessorST : owns via thread
```

**`ImageProcessor` Dispatch Priority**: OpenGL (GPU-accelerated) → G2D (if supported for the format pair) → CPU (general fallback). Environment variables `EDGEFIRST_DISABLE_GL`, `EDGEFIRST_DISABLE_G2D`, `EDGEFIRST_DISABLE_CPU` can override this chain.

**Supported Operations**:
- Format conversion (YUYV, VYUY, NV12, NV16, RGB, RGBA, BGRA, GREY, Planar RGB, Planar RGBA, RGB int8, Planar RGB int8)
- Resize with various interpolation methods
- Rotation (0°, 90°, 180°, 270°)
- Flip (horizontal, vertical)
- Crop and region-of-interest
- Instance segmentation mask rendering (draw and decode workflows)

**TensorDyn (Image Representation)**:

`TensorDyn` is the type-erased tensor enum that serves as the primary image type.
It wraps `Tensor<T>` variants for different element types and carries an optional
`PixelFormat` describing the spatial pixel layout. The format and element type are
orthogonal: `PixelFormat` describes the spatial arrangement (e.g. packed RGB,
semi-planar NV12) while `DType` describes the element storage (e.g. `U8`, `I8`,
`F32`).

```rust
pub struct Tensor<T> {
    storage: TensorStorage<T>,
    format: Option<PixelFormat>,
    /// Second plane for multiplane NV12/NV16 (separate DMA-BUF allocation).
    chroma: Option<Box<Tensor<T>>>,
}
```

The `chroma` field supports multi-plane DMA-BUF NV12/NV16 where Y and UV planes
are in separate allocations (common with V4L2 `NV12M` format). Constructed via
`Tensor::from_planes()`. See [Appendix A: Multi-Plane DMA-BUF](#appendix-a-multi-plane-dma-buf-limitation) for details.

Image construction uses `TensorDyn::image(w, h, PixelFormat, DType, mem)` which
allocates the appropriate tensor shape and sets the pixel format metadata.

Width, height, channels, and stride are **not stored** — they are computed from the tensor shape and `PixelFormat` on every access. The tensor shape encoding depends on the pixel format:

| Format | Tensor Shape | Notes |
|--------|-------------|-------|
| Rgb, Rgba, Bgra, Grey, Yuyv, Vyuy | `[H, W, C]` | Interleaved (channels-last) |
| PlanarRgb, PlanarRgba | `[C, H, W]` | Channels-first |
| Nv12 | `[H*3/2, W]` | 2D — Y plane (H rows) + UV plane (H/2 rows) |
| Nv16 | `[H*2, W]` | 2D — Y plane (H rows) + UV plane (H rows) |

**PixelFormat Enum**:

Pixel formats are variants of the `PixelFormat` enum defined in the tensor crate.
Int8 variants use `DType::I8` with any `PixelFormat` rather than separate format constants.

| Variant | Display | Channels | Layout |
|---------|---------|----------|--------|
| `PixelFormat::Rgb` | RGB | 3 | Packed |
| `PixelFormat::Rgba` | RGBA | 4 | Packed |
| `PixelFormat::Bgra` | BGRA | 4 | Packed |
| `PixelFormat::Grey` | Y800 | 1 | Packed |
| `PixelFormat::Yuyv` | YUYV | 2 | Packed |
| `PixelFormat::Vyuy` | VYUY | 2 | Packed |
| `PixelFormat::Nv12` | NV12 | 1 | SemiPlanar |
| `PixelFormat::Nv16` | NV16 | 1 | SemiPlanar |
| `PixelFormat::PlanarRgb` | PlanarRgb | 3 | Planar |
| `PixelFormat::PlanarRgba` | PlanarRgba | 4 | Planar |

**Planar RGB Format**:
Planar RGB (`PixelFormat::PlanarRgb`) stores color channels in separate planes rather than interleaved. This format is particularly useful for:
- Neural network preprocessing where planar layout is required
- Hardware accelerators that prefer planar data
- Efficient SIMD operations on individual color channels
- GPU texture operations via OpenGL with swizzled grayscale textures

**Image Processing Flow**:
```mermaid
flowchart TD
    Input[Input Image<br/>JPEG/PNG bytes or raw pixels]
    TI[TensorDyn<br/>type-erased tensor + PixelFormat]
    Conv{ImageProcessor::convert<br/>Backend selection}
    G2D[G2D Acceleration<br/>NXP i.MX only]
    GL[OpenGL Acceleration<br/>GPU accelerated]
    CPU[CPU Fallback<br/>fast_image_resize]
    Output[Output Image<br/>TensorDyn or numpy array]
    
    Input --> TI
    TI --> Conv
    Conv -->|Supported on i.MX| G2D
    Conv -->|Linux with GPU| GL
    Conv -->|Always available| CPU
    G2D --> Output
    GL --> Output
    CPU --> Output
    
    style TI fill:#e1f5ff
    style Conv fill:#fff4e1
    style G2D fill:#90ee90
    style GL fill:#87ceeb
    style CPU fill:#ffeb9c
    style Output fill:#e8f5e9
```

**GPU-Optimal Image Creation (`ImageProcessor::create_image()`)**:

`create_image()` is the preferred way to allocate images for use with
`ImageProcessor::convert()`. It probes the GPU at initialization time and
selects the best available memory backend in priority order:

```mermaid
flowchart TD
    Create["ImageProcessor::create_image(w, h, PixelFormat, DType, mem)"]
    DMA{DMA-buf roundtrip<br/>verified at init?}
    PBO{OpenGL PBO<br/>available?}
    Mem[MemTensor<br/>heap fallback]

    Create --> DMA
    DMA -->|Yes| UseDMA["DmaTensor<br/>Zero-copy EGLImage import"]
    DMA -->|No| PBO
    PBO -->|Yes| UsePBO["PboTensor<br/>Zero-copy GL buffer binding"]
    PBO -->|No| Mem

    style UseDMA fill:#90ee90
    style UsePBO fill:#87ceeb
    style Mem fill:#ffeb9c
```

| Backend | When selected | GPU transfer method | Platforms |
|---------|---------------|---------------------|-----------|
| DMA-buf | GPU supports EGLImage import from DMA-buf FDs | Zero-copy: `EGL_EXT_image_dma_buf_import` — the GPU reads/writes the DMA buffer directly via EGLImage, no pixel copies | NXP i.MX 8M Plus (Vivante), NXP i.MX 95 (Mali/Panfrost) |
| PBO | OpenGL ES 3.0 available but DMA-buf roundtrip fails | Zero-copy GL binding: `GL_PIXEL_UNPACK_BUFFER` for upload, `GL_PIXEL_PACK_BUFFER` for readback — data stays in GPU-accessible memory | NVIDIA desktop GPUs, systems without DMA-buf permissions |
| Mem | No GPU or OpenGL not available | CPU `memcpy` via `glTexImage2D`/`glReadnPixels` with mapped host pointers | All platforms (fallback) |

**GL Transfer Backend Selection**:

The OpenGL processor selects a transfer backend at initialization time:

| Backend | Detection | GPU Upload | GPU Readback |
|---------|-----------|-----------|--------------|
| `DmaBuf` | `verify_dma_buf_roundtrip()` passes | `EGL_EXT_image_dma_buf_import` (zero-copy) | EGLImage export (zero-copy) |
| `Pbo` | OpenGL ES 3.0 available, DMA-buf fails | `GL_PIXEL_UNPACK_BUFFER` | `GL_PIXEL_PACK_BUFFER` |
| `Sync` | Fallback | `glTexImage2D` with host pointer | `glReadnPixels` to host pointer |

**GL Thread Architecture**:

`GLProcessorThreaded` is the public thread-safe wrapper. It spawns a dedicated GL thread that owns the EGL context and all GL state (`GLProcessorST`). All operations are sent as `GLProcessorMessage` enum variants through a channel and block on a oneshot reply. This design is required because EGL contexts are thread-local — all GL calls must happen on the thread that created the context.

The `PboOps` trait bridges the tensor crate and the GL thread: `PboTensor` holds a `WeakSender` to the GL thread channel. When the tensor needs to map/unmap/delete the PBO, it sends a message through this channel. The weak sender ensures PBO tensors don't prevent GL thread shutdown.

**Why PBO matters**: On desktop Linux with NVIDIA GPUs, DMA-buf allocation
succeeds (via `/dev/dma_heap/system`) but the NVIDIA EGL driver cannot import
those buffers — the `verify_dma_buf_roundtrip()` check catches this at
initialization. Without PBO, every `convert()` call would fall back to CPU
`memcpy` for upload and readback. PBO keeps the data in GPU-accessible buffers,
enabling the same zero-copy shader pipeline used on DMA platforms.

**PBO Convert Dispatch**:

When `convert()` is called with PBO-backed images, the GL thread must not call
`tensor.map()` on those images — doing so would send a message back to itself
and deadlock. Instead, the convert path dispatches to specialized methods:

```
┌─────────────────────────────┬──────────────────────────────────────────┐
│ Source → Destination        │ Method                                    │
├─────────────────────────────┼──────────────────────────────────────────┤
│ DMA → DMA                   │ convert_dest_dma (EGLImage both sides)   │
│ PBO → PBO                   │ convert_pbo_to_pbo (UNPACK + PACK)       │
│ Mem/DMA → PBO               │ convert_any_to_pbo (texture + PACK)      │
│ PBO → Mem                   │ convert_pbo_to_mem (UNPACK + ReadnPixels) │
│ Mem/DMA → Mem/DMA           │ convert_dest_non_dma (texture + memcpy)  │
└─────────────────────────────┴──────────────────────────────────────────┘
```

**EGL Image Cache**:

The OpenGL backend maintains two independent LRU caches of EGLImages — one for
source tensors (`src_egl_cache`) and one for destination tensors
(`dst_egl_cache`). Each entry is keyed by `(BufferIdentity.id, chroma_id)`.

`BufferIdentity` is a small struct that pairs a globally unique monotonic ID
(allocated from an `AtomicU64` counter) with an `Arc<()>` liveness guard:

```
BufferIdentity {
    id:    u64,     // unique per allocation — new value each time from_fd() is called
    guard: Arc<()>, // cache entries hold Weak<()>; when tensor drops, weak dies
}
```

When a tensor is freed, its `Arc<()>` guard drops. The cache holds only a
`Weak<()>` reference, so `sweep()` detects dead entries without requiring an
explicit removal call.

| Pattern | Cache behavior | Performance |
|---------|---------------|-------------|
| Same tensor object reused across frames | Cache hit on every frame | Fast path — no EGLImage re-import |
| New tensor created from same fd each frame | Cache miss on every frame | Slow path — EGLImage re-imported each call |
| `dst` of call N reused as `src` of call N+1 | Hit in both caches | Two separate entries, no collision |

**Key design implications**:

- `hal_tensor_from_fd()` and `hal_import_image()` always allocate a new
  `BufferIdentity`. Callers that re-wrap the same fd each frame will see a cache
  miss on every `convert()` call. Hold the tensor object alive across frames.

- Content written into a DMA-BUF between `convert()` calls (e.g., by a V4L2
  decoder) is visible on the next call. EGLImage is a handle to live physical
  memory — it is not a snapshot. The tensor wrapper does not need to be recreated
  when the buffer contents change.

- The `last_bound_src_egl` optimization skips the `glEGLImageTargetTexture2DOES`
  rebind when the source EGLImage has not changed between consecutive calls. This
  is safe because the texture already points at the correct DMA-BUF memory; skipping
  the rebind does not prevent new frame data from being read.

- A `glFinish()` is issued at the end of every `convert()` call. This guarantees
  that all GPU reads of the source and writes to the destination are complete
  before the function returns, making it safe to chain calls and to let the caller
  write new data into a source buffer immediately after `convert()` returns.

**Vivante NV12 to Planar RGB Two-Pass Workaround**:

A single-pass NV12 → PlanarRgb shader causes a GPU hang on the Vivante GC7000UL
(NXP i.MX 8M Plus). The workaround splits the conversion into two passes:

```
Pass 1:  NV12  →  RGBA  (intermediate)
         All geometry operations here: resize, crop, rotation, flip, letterbox

Pass 2:  RGBA  →  PlanarRgb  (at destination resolution)
         Deinterleaves RGBA to three separate planes using sampler2D variants
```

Pass 1 reuses the existing `packed_rgb_intermediate_tex` GL texture — no new
GPU resources are allocated. Pass 2 uses the same shader infrastructure as
direct RGBA → PlanarRgb conversions. The two-pass path is selected automatically
when `is_vivante && src_fmt == Nv12 && dst_fmt.layout() == Planar`. No API
changes are required from callers.

### 3. Decoder HAL (`edgefirst_decoder`)

**Purpose**: Post-processing for object detection and segmentation model outputs.

**Supported Decoders**:
- **YOLO** (YOLOv5, YOLOv8, YOLOv11, YOLOv26)
  - Object detection
  - Instance segmentation
  - Split output format support
  - End-to-end models with embedded NMS (YOLOv26)
  - Mixed data type support (different types per input tensor)
- **ModelPack** (Au-Zone proprietary format)
  - Detection with anchor-based decoding

**YOLO26 End-to-End Support**:

YOLOv26 models can be exported with embedded NMS (one-to-one matching heads),
eliminating external NMS post-processing. The `DecoderVersion::Yolo26` variant
triggers end-to-end model type selection:

| Config Field | Value | Effect |
|-------------|-------|--------|
| `decoder_version` | `"yolo26"` | Selects end-to-end model types, bypasses NMS |
| `decoder_version` | `"yolov8"` | Uses traditional model types with external NMS |

When `decoder_version` is `"yolo26"`, `DecoderVersion::is_end_to_end()` returns
`true` and the decoder selects one of the end-to-end `ModelType` variants:

| ModelType | Tensors | Format |
|-----------|---------|--------|
| `YoloEndToEndDet` | 1 | `[batch, N, 6+]` — xyxy, conf, class |
| `YoloEndToEndSegDet` | 2 | `[batch, N, 6+num_protos]` + protos |
| `YoloSplitEndToEndDet` | 3 | boxes `[B,N,4]` + scores `[B,N,1]` + classes `[B,N,1]` |
| `YoloSplitEndToEndSegDet` | 5 | boxes + scores + classes + mask_coeff + protos |

For non-end-to-end YOLO26 exports (`end2end=false`), use `decoder_version: "yolov8"`
with an explicit `nms` field (`ClassAgnostic` or `ClassAware`).

**NMS Modes** (`Option<Nms>`):
- `Some(Nms::ClassAgnostic)` — suppress overlapping boxes regardless of class (default)
- `Some(Nms::ClassAware)` — only suppress boxes sharing the same class label
- `None` — bypass NMS entirely (auto-set for end-to-end models)

**Proto Mask API**:

For segmentation models, `decode_quantized_proto()` and `decode_float_proto()`
return raw proto data and mask coefficients without materializing pixel masks.
These are the preferred entry point for fused GPU rendering via
`ImageProcessor::draw_masks_proto()`.

**Architecture**:
```mermaid
flowchart LR
    Builder[DecoderBuilder]
    Decoder[Decoder]
    Det[decode_detection<br/>→ bboxes, scores, classes]
    Seg[decode_segmentation<br/>→ bboxes, scores, classes, masks]
    
    Builder --> Decoder
    Decoder --> Det
    Decoder --> Seg
    
    style Builder fill:#e1f5ff
    style Decoder fill:#fff4e1
    style Det fill:#e8f5e9
    style Seg fill:#e8f5e9
```

**Detection Pipeline**:
```mermaid
flowchart TD
    Raw[Model Raw Output<br/>quantized or float]
    E2E{End-to-end?<br/>decoder_version = yolo26}

    Quant{Quantized?}
    Dequant[Dequantization<br/>scale, zero_point]

    Parse[Parse boxes & scores<br/>XYWH → XYXY conversion]
    NMS[Non-Maximum Suppression<br/>IoU threshold filtering]
    Filter[Filter by score threshold]

    E2EParse[Parse post-NMS output<br/>XYXY + conf + class directly]
    E2EFilter[Filter by score threshold]

    Det[Detection boxes<br/>bbox, score, class]
    Seg[Segmentation masks<br/>per-box mask matrices]

    Raw --> E2E
    E2E -->|Yes| E2EParse
    E2EParse --> E2EFilter
    E2EFilter --> Det
    E2EFilter --> Seg

    E2E -->|No| Quant
    Quant -->|Yes| Dequant
    Quant -->|No| Parse
    Dequant --> Parse
    Parse --> NMS
    NMS --> Filter
    Filter --> Det
    Filter --> Seg

    style Raw fill:#e1f5ff
    style E2E fill:#fff4e1
    style Dequant fill:#fff4e1
    style NMS fill:#ffeb9c
    style E2EParse fill:#87ceeb
    style Det fill:#90ee90
    style Seg fill:#90ee90
```

#### Instance Segmentation Mask Rendering

YOLO segmentation models produce **proto masks** (shared basis masks at reduced
resolution, typically 160x160) and **mask coefficients** (per-detection linear
combination weights). The raw mask for detection `i` is:

```
mask_raw[i] = coefficients[i] @ protos    # shape: (proto_h, proto_w)
```

The HAL provides three workflows for consuming these masks:

| Workflow | Python | Rust | C | CPU | OpenGL | G2D |
|----------|--------|------|---|:---:|:------:|:---:|
| **Decode** — per-detection binary masks | `decoder.decode_masks()` | `decode_masks_atlas()` | `hal_decoder_decode_masks()` | Yes | Yes | No |
| **Draw** — fused overlay onto image | `decoder.draw_masks()` | `draw_masks_proto()` | `hal_decoder_draw_masks()` | Yes | Yes | No |
| **Draw pre-decoded** — draw already-decoded masks | `processor.draw_masks()` | `draw_masks()` | `hal_image_processor_draw_masks()` | Yes | Yes | No |

> **G2D limitation:** The NXP G2D hardware accelerator does not support mask
> rendering. On platforms where G2D is the primary image processor (e.g.
> i.MX 8M Plus without EGL), all mask methods return `NotImplemented`. Use
> an OpenGL-capable `ImageProcessor` (pass an `egl_display`) or fall back
> to CPU rendering.

**Choosing between `decode_masks` and `draw_masks`:**

| Use case | Recommended API | Why |
|----------|----------------|-----|
| Overlay colored masks onto a display frame | `decoder.draw_masks()` | Fused path — masks never leave Rust/GPU, lowest latency |
| Export per-instance binary masks for downstream processing (tracking, area measurement, custom compositing) | `decoder.decode_masks()` | Returns individual `uint8` arrays you can manipulate in Python/C |
| Draw masks you already have (e.g. from a previous `decode()` call) | `processor.draw_masks()` | Accepts pre-decoded `(H, W, C)` mask arrays |

**Format requirements:**

- **CPU backend:** destination image must be `RGBA` or `RGB`.
- **OpenGL backend:** destination image must be `RGBA`, `BGRA`, or `RGB`.
- **`decode_masks`** does not require a destination image (masks are returned as arrays). The `output_width` and `output_height` parameters define the **coordinate space** for interpreting bounding boxes — they are not the dimensions of the returned mask arrays. Each returned mask is sized to its detection's bounding box (`bbox_h × bbox_w` pixels), containing binary `uint8` values where `255` = mask presence and `0` = background.

**Performance characteristics (YOLOv8n-seg, 640×640, ~5 detections):**

| Platform | `draw_masks` (fused) | `decode_masks` (atlas) | Notes |
|----------|---------------------|----------------------|-------|
| **i.MX 8M Plus (imx8mp-frdm)** | ~8–12 ms (OpenGL) | ~10–15 ms (OpenGL) | Vivante GC7000UL GPU; PBO readback adds latency for atlas path |
| **i.MX 8M Plus (CPU only)** | ~25–40 ms | ~20–35 ms | Single-core Cortex-A53; scales linearly with detection count and bbox area |
| **i.MX 95 (imx95-frdm)** | ~4–7 ms (OpenGL) | ~5–9 ms (OpenGL) | Mali G310 (Panfrost); faster PBO readback than i.MX 8M Plus |
| **x86_64 desktop (CPU)** | ~3–5 ms | ~2–4 ms | For development; not representative of target hardware |

> These are representative ranges for typical COCO-class detections. Actual
> timings depend on detection count, bounding box sizes, and proto tensor
> quantization format. Use `mask_benchmark` for precise on-target measurements:
> `cargo bench -p edgefirst-image --bench mask_benchmark`

**Fused proto→pixel algorithm (`draw_masks_proto`)**

Instead of computing the matmul at proto resolution and upsampling the result,
the fused path upsamples the proto field itself and evaluates the dot product at
every output pixel:

```
For each output pixel (x, y) in bbox at 640×640:
    bilinear_sample(protos, proto_coords(x, y))  →  32 interpolated values
    dot(coefficients, interpolated_protos)        →  raw logit
    sigmoid(raw)                                  →  mask value [0, 1]
    threshold at 0.5 → blend color onto pixel (draw path)
    — or —
    threshold at 0.0 on logit → 0/255 uint8 pixel (decode/atlas path)
```

This is algebraically equivalent to bilinear upsampling after matmul (because
both bilinear interpolation and the dot product are linear), but avoids
materializing intermediate tensors. Key design choices:

- **No proto-resolution crop** — the full 160×160 proto field is sampled,
  avoiding the boundary erosion artifact of crop-before-upsample approaches.
- **Sigmoid after interpolation** — sigmoid is nonlinear, so applying it after
  spatial operations preserves the full dynamic range through interpolation.
- **Binary output for decode path** — both CPU and GPU atlas paths produce
  binary `0`/`255` uint8 masks (thresholded at sigmoid 0.5 or logit 0.0).
  The draw path uses the sigmoid value for alpha-blend weighting.

This approach is mathematically equivalent to Ultralytics' `retina_masks=True`
(`process_mask_native`) for binary mask output. Empirical validation across 26
matched detections on COCO val2017 images confirms **0.993 mean mask IoU**
between the two methods.

**GPU implementation (OpenGL)**

Two shader families are used depending on the workflow:

*Draw path (`draw_masks_proto`) — sigmoid shaders with alpha blending:*

The fragment shader computes sigmoid(logit) and blends the detection color onto
the framebuffer using `GL_SRC_ALPHA / GL_ONE_MINUS_SRC_ALPHA`.

*Atlas/decode path (`decode_masks_atlas`) — logit-threshold shaders:*

| Shader | Proto format | Interpolation | Notes |
|--------|-------------|---------------|-------|
| `logit_int8_nearest` | R8I (quantized) | Nearest | Fastest, lowest quality |
| `logit_int8_bilinear` | R8I (quantized) | Manual bilinear in shader | Manual 4-tap with dequantization |
| `logit_f32` | R32F (float) | Hardware `texture()` with GL_LINEAR | Best quality, uses GPU sampler |

These shaders output binary `logit > 0 ? 1.0 : 0.0` (skipping the `exp()` per
fragment), which the PBO readback maps to uint8 `0`/`255`.

The GPU renders a quad per detection, the fragment shader evaluates the mask at
every pixel, and the result is read back via PBO as R8 uint8 values. For the
atlas path, all detections are packed into a single texture atlas and read back
in one PBO transfer.

### 4. Tracker HAL (`edgefirst_tracker`)

**Purpose**: Multi-object tracking across video frames.

**Implementation**: ByteTrack algorithm with Kalman filtering

**Architecture**:
```mermaid
classDiagram
    class ByteTrack {
        +update(detections) TrackInfo[]
    }
    
    class Tracklet {
        +UUID uuid
        +KalmanFilter filter
        +int track_count
        +Timestamps timestamps
    }
    
    ByteTrack *-- Tracklet : manages
```

**Tracking Flow**:
```mermaid
flowchart TD
    NewFrame[New Frame Detections]
    Predict[Predict tracklet positions<br/>Kalman filter forward step]
    Cost[Compute cost matrix<br/>IoU between predictions and detections]
    Hungarian[Hungarian Algorithm LAPJV<br/>optimal assignment]
    
    Matched[Matched:<br/>Update tracklet]
    UnmatchedDet[Unmatched detection:<br/>Create new tracklet]
    UnmatchedTrack[Unmatched tracklet:<br/>Increment lost count]
    
    CheckLost{Lost > threshold?}
    Delete[Delete tracklet]
    Keep[Keep tracklet]
    
    NewFrame --> Predict
    Predict --> Cost
    Cost --> Hungarian
    Hungarian --> Matched
    Hungarian --> UnmatchedDet
    Hungarian --> UnmatchedTrack
    UnmatchedTrack --> CheckLost
    CheckLost -->|Yes| Delete
    CheckLost -->|No| Keep
    
    style Matched fill:#90ee90
    style UnmatchedDet fill:#87ceeb
    style Delete fill:#ffcccb
```

### 5. Python Bindings (`edgefirst-hal`)

**Purpose**: Expose HAL functionality to Python via PyO3.

**Exposed Classes**:
- `Tensor`: Unified tensor with image support and numpy buffer protocol
- `ImageProcessor`: Image processing operations
- `Decoder`: Model output decoding
- `PixelFormat`, `Normalization`, `Rect`, `Rotation`, `Flip`: Configuration types

**Python Integration**:
```mermaid
flowchart LR
    Py[Python Code]
    PyO3[PyO3 Bindings]
    Rust[Rust Core HAL]
    HW[Hardware Accelerators / CPU]

    Py --> PyO3
    PyO3 --> Rust
    Rust --> HW

    style Py fill:#3776ab,color:#fff
    style PyO3 fill:#ce422b
    style Rust fill:#dea584
    style HW fill:#90ee90
```

### Cross-Crate Dependencies

The `edgefirst_image` crate depends on `edgefirst_decoder` for the `DetectBox`, `ProtoData`, and `Segmentation` types used in the mask rendering APIs (`draw_masks`, `draw_masks_proto`, `decode_masks_atlas`). This means the image crate imports decoder types but does not import the `Decoder` itself — it only needs the output data structures that describe detections and masks.

### 6. C API Bindings (`edgefirst-hal-capi`)

**Purpose**: Expose HAL functionality to C/C++ consumers via cbindgen-generated headers.

**Architecture**:
- Builds as both `staticlib` and `cdylib`
- cbindgen generates C headers from Rust source annotations
- Opaque handle pattern: C code operates on `HalTensor*`, `HalImageProcessor*`, etc.
- All functions return error codes; error messages retrieved via `hal_error_message()`
- Covers tensor, image, decoder, and tracker APIs

**Source files** (`crates/capi/src/`):

| File | Lines | Scope |
|------|------:|-------|
| `tensor.rs` | ~1,200 | Tensor create, map, reshape, fd sharing |
| `image.rs` | ~1,800 | ImageProcessor, convert, draw |
| `decoder.rs` | ~2,800 | Decoder create, decode detection/segmentation |
| `tracker.rs` | ~300 | ByteTrack create, update |
| `error.rs` | ~120 | Error handling utilities |

### 7. G2D FFI (`g2d-sys`)

**Purpose**: Foreign Function Interface to NXP i.MX G2D library.

**Architecture**:
- Raw FFI bindings via bindgen
- Safe Rust wrapper types
- IOCTL interface for DMA buffer operations
- Version detection and capability queries

### 8. GPU Probe (`gpu-probe`)

**Purpose**: Standalone binary for probing GPU capabilities and benchmarking transfer methods.

Used during development and CI to verify EGL/OpenGL support, benchmark RGB packing strategies, test DMA-buf roundtrip, and measure pipeline throughput on target hardware.

## Common API Usage Patterns

### Pattern 1: Basic Image Conversion

```rust
use edgefirst_image::{load_image, ImageProcessor, ImageProcessorTrait, Rotation, Flip, Crop};
use edgefirst_tensor::{PixelFormat, DType, TensorDyn};

// Load image from JPEG
let bytes = std::fs::read("testdata/zidane.jpg")?;
let input = load_image(&bytes, Some(PixelFormat::Rgb), None)?;

// Create converter (auto-selects best backend: G2D, OpenGL, CPU)
let mut converter = ImageProcessor::new()?;

// Create output buffer with optimal GPU memory (DMA > PBO > Mem)
let mut output = converter.create_image(640, 640, PixelFormat::Rgb, DType::U8, None)?;

// Convert and resize — zero-copy if DMA or PBO backend is active
converter.convert(&input, &mut output, Rotation::None, Flip::None, Crop::default())?;
```

### Pattern 2: Detection Decoding

```rust
use edgefirst_hal::decoder::Decoder;
use std::collections::HashMap;

// Build decoder from configuration dictionary/JSON
let config: HashMap<String, serde_json::Value> = 
    serde_json::from_str(&config_json)?;

let decoder = Decoder::new(config, 0.5, 0.45)?;  // score_thresh, iou_thresh

// Decode model outputs (supports mixed types per tensor)
let outputs = vec![boxes_tensor, scores_tensor];
let (bboxes, scores, classes) = decoder.decode_detection(&outputs)?;
```

**Python Example**:
```python
import edgefirst_hal
import numpy as np

# Create decoder from config dict or YAML
decoder = edgefirst_hal.Decoder(config_dict, 0.5, 0.45)

# Decode outputs (automatically handles quantization)
boxes, scores, classes = decoder.decode([output0, output1])
```

### Pattern 3: Multi-Frame Tracking

```rust
use edgefirst_hal::tracker::{ByteTrack, Tracker, DetectionBox};

let mut tracker = ByteTrack::default();

for frame in video_frames {
    let detections = run_detection(frame)?;
    let track_infos = tracker.update(&detections, frame.timestamp);
    
    for track_info in track_infos {
        println!("Object {}: {:?}", track_info.uuid, track_info.tracked_location);
    }
}
```

### Pattern 4: Zero-Copy Tensor Sharing

```rust
use edgefirst_hal::tensor::{Tensor, TensorTrait};

// Create tensor in process A
let tensor = Tensor::<u8>::new(&[1920, 1080, 3], None, Some("frame1"))?;
let fd = tensor.clone_fd()?;

// Send fd to process B (via Unix domain socket, etc.)
// ...

// Process B recreates tensor from fd
let shared_tensor = Tensor::<u8>::from_fd(fd, &[1920, 1080, 3], None)?;
```

### Pattern 5: Python API Usage

```python
import edgefirst_hal as ef
import numpy as np

# Load image from file
tensor_img = ef.Tensor.load("testdata/zidane.jpg", ef.PixelFormat.Rgb)

# Create converter
converter = ef.ImageProcessor()

# Create output image with optimal GPU memory (DMA > PBO > Mem)
output = converter.create_image(640, 640, ef.PixelFormat.Rgb)

# Resize with hardware acceleration
converter.convert(tensor_img, output)

# Convert to numpy for processing
output_array = np.zeros((640, 640, 3), dtype=np.uint8)
output.normalize_to_numpy(output_array)
```

## Design Patterns

### 1. Trait-Based Polymorphism

The HAL uses Rust traits extensively to provide polymorphic behavior:
- `TensorTrait<T>`: Common interface for all tensor types
- `ImageProcessorTrait`: Common interface for all image converters
- `DetectionBox`: Trait for objects with bounding boxes

### 2. Enum Dispatch

Uses `enum_dispatch` crate for zero-cost polymorphism without dynamic dispatch overhead.

### 3. Builder Pattern

Complex objects use the builder pattern (e.g., `DecoderBuilder`) for flexible construction.

### 4. Zero-Copy Operations

Extensive use of:
- Memory-mapped file descriptors
- Slice views into tensors
- ndarray views for array operations

### 5. Hardware Fallback Chain

Operations try hardware accelerators first, falling back to CPU implementations gracefully.

### 6. Type-Safe Foreign Interfaces

Raw FFI bindings are wrapped in safe Rust types that enforce correct usage at compile time.

### 7. Python Wrapper Naming Convention

Python wrapper types use a `Py` prefix (e.g., `PyTensor`, `PyPixelFormat`) to clearly distinguish them from their Rust counterparts. The Python `Tensor` class wraps `TensorDyn` internally. This convention makes it explicit which types are Python-facing and which are internal Rust types.

## EGL/GL Resource Cleanup at Process Shutdown

### Problem

The OpenGL headless renderer (`crates/image/src/opengl_headless.rs`) uses EGL and
OpenGL ES via dynamically loaded shared libraries (`libEGL.so.1`, `libGLESv2.so`).
When the process exits — particularly from a Python interpreter running PyO3
extensions — EGL resource cleanup can crash with heap corruption, segfaults, or
panics. This is a well-documented, industry-wide problem with no clean solution.

The crash occurs due to a fundamental conflict between four systems during
process shutdown:

1. **Python finalization order is non-deterministic.** During `Py_FinalizeEx()`,
   Python destroys modules and objects in random order. A PyO3 `#[pyclass]`
   wrapping `GlContext` may have its Rust `Drop` invoked after dependent state
   has already been torn down.

2. **Linux `atexit` handler ordering is unreliable.** The glibc `__cxa_finalize`
   mechanism interacts with `dlclose` in ways that produce non-deterministic
   ordering between atexit handlers registered by different shared libraries.
   A known glibc bug ([glibc #21032](https://sourceware.org/bugzilla/show_bug.cgi?id=21032))
   prevents proper cleanup of TLS destructors from unloaded libraries.

3. **Mesa's `_eglAtExit` use-after-free.** Mesa registers an atexit handler
   that frees per-thread EGL state (`_EGLThreadInfo`). If our `Drop` calls
   `eglReleaseThread()` after this handler has already run, it dereferences
   freed memory ([Ubuntu Bug #1946621](https://bugs.launchpad.net/ubuntu/+source/mesa/+bug/1946621)).

4. **Vendor EGL driver bugs.** Some vendor EGL drivers (e.g. Qualcomm Adreno,
   older NVIDIA) may misbehave during cleanup. On NXP Vivante, earlier
   observations suggested a double-free in `gcoOS_FreeMemo` when both
   `eglDestroyContext` and `eglTerminate` are called, but this was not
   conclusively confirmed. The `catch_unwind` guard (Layer 3) absorbs any
   driver-side panics during cleanup.

### Solution

The implementation uses a defense-in-depth strategy with four layers, each
addressing a different failure mode:

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Box::leak — EGL library handle                 │
│   Prevents dlclose from unmapping shared library code   │
├─────────────────────────────────────────────────────────┤
│ Layer 2: ManuallyDrop<Rc<Egl>> — EGL instance           │
│   Prevents khronos-egl Drop from calling                │
│   eglReleaseThread() into freed Mesa state              │
├─────────────────────────────────────────────────────────┤
│ Layer 3: catch_unwind — EGL cleanup calls               │
│   Catches panics from eglDestroyContext/eglMakeCurrent  │
│   if function pointers are invalidated                  │
├─────────────────────────────────────────────────────────┤
│ Layer 4: eglTerminate inside catch_unwind               │
│   Releases ref-counted display; catch_unwind absorbs    │
│   any driver-side misbehaviour                          │
└─────────────────────────────────────────────────────────┘
```

The `GlContext::drop` implementation performs explicit EGL resource cleanup
(destroy context, destroy surface, terminate display) inside `catch_unwind`,
then intentionally skips dropping the `Rc<Egl>` wrapper. The EGL library
handle is leaked via `Box::leak` at load time so it is never `dlclose`'d.
`eglTerminate` is ref-counted per the EGL spec: each `eglInitialize`
increments a reference count and each `eglTerminate` decrements it, so the
display is only truly torn down when the last reference is released.

### Industry Precedent

This approach is consistent with how other major graphics projects handle the
same problem:

- **Chromium/ANGLE** skips full EGL teardown on GPU process exit, treating
  cleanup as more dangerous than no cleanup during shutdown.

- **wgpu** (the Rust WebGPU implementation used by Firefox) wraps its
  `glow::Context` in `ManuallyDrop` and loads EGL with `RTLD_NODELETE` to
  prevent library unloading. The `khronos-egl` crate itself adopted
  `RTLD_NOW | RTLD_NODELETE` on Linux after
  [khronos-egl #14](https://github.com/timothee-haudebourg/khronos-egl/issues/14),
  referencing the same glibc root cause.

- **Smithay** (Wayland compositor toolkit) supports skipping `eglTerminate`
  via a feature flag for the same class of driver bugs.

### Limitations

`catch_unwind` catches Rust panics but cannot catch fatal signals (`SIGABRT`,
`SIGSEGV`, `SIGBUS`) that originate from heap corruption inside the C driver.
The `Box::leak` layer prevents `dlclose` from unmapping driver code, and the
`ManuallyDrop<Rc<Egl>>` layer avoids `eglReleaseThread` calls into freed
Mesa state. If a driver causes a fatal signal during `eglTerminate`, the
process will crash — but this has not been observed in practice on any
supported platform (NXP i.MX8 Vivante, NXP i.MX95 Mali/Panfrost, Mesa x86_64).

### References

- Mesa atexit use-after-free: [Ubuntu Bug #1946621](https://bugs.launchpad.net/ubuntu/+source/mesa/+bug/1946621)
- glibc TLS destructor bug: [glibc #21032](https://sourceware.org/bugzilla/show_bug.cgi?id=21032)
- khronos-egl RTLD_NODELETE fix: [khronos-egl #14](https://github.com/timothee-haudebourg/khronos-egl/issues/14)
- wgpu SIGSEGV on library unload: [wgpu #246](https://github.com/gfx-rs/wgpu/issues/246)
- PyO3 Drop after finalization: [PyO3 #4632](https://github.com/PyO3/pyo3/issues/4632)
- Python finalization order: [CPython docs — Py_FinalizeEx](https://docs.python.org/3/c-api/init.html)

## G2D Resource Cleanup

### Problem

On NXP i.MX platforms, the G2D hardware accelerator (`libg2d.so.2`) and the EGL
library (`libEGL.so.1`) share kernel driver state through the Vivante `galcore`
device (`/dev/galcore`). When both libraries are loaded, calling `dlclose` on
either one can trigger heap corruption (`corrupted double-linked list`) during
process exit because the atexit handlers registered by the shared `galcore`
driver become inconsistent.

### Solution

For production code, `G2DProcessor` is used normally and dropped by Rust's
ownership system. The `Box::leak` of the EGL library handle (Layer 1 in the
EGL cleanup strategy above) prevents `dlclose` from unmapping `libEGL.so.1`,
which keeps the shared `galcore` state intact during process shutdown.

For benchmark code (where `G2DProcessor` instances are created and destroyed
many times within a single process), use `ManuallyDrop<G2DProcessor>` to
prevent repeated `g2d_close` + `dlclose` cycles that can exhaust driver
resources. The benchmark harness (`crates/bench`) wraps G2D processor
instances in `ManuallyDrop` to avoid this pattern.

### Resource Leak Prevention Policy

All GPU and DMA resources must be properly released. Specifically:

- **EGL displays**: `eglTerminate` must be called in `GlContext::drop`. The
  EGL spec ref-counts display connections, so each `eglTerminate` decrements
  the count and only tears down state when it reaches zero.
- **EGL contexts**: `eglDestroyContext` must be called before `eglTerminate`.
  No EGL surfaces are created — the HAL uses surfaceless contexts
  (`EGL_KHR_surfaceless_context` + `EGL_KHR_no_config_context`) and renders
  exclusively through FBOs backed by EGLImages imported from DMA-buf.
- **DMA buffers**: Closed via file descriptor `close()` in `Drop`.
- **G2D contexts**: `g2d_close` via `G2DProcessor::drop`.

Intentional leaks are limited to the EGL **library handle** (`Box::leak` to
prevent `dlclose`) and the `Rc<Egl>` wrapper (`ManuallyDrop` to prevent
`eglReleaseThread`). These are lightweight Rust-side objects; the actual
GPU/display resources are released by the explicit cleanup calls above.

## Testing with GPU Resources

### Single-Threaded Test Execution

All test execution must use single-threaded mode (`--test-threads=1` for
`cargo test`, `-j 1` for `cargo nextest`). This is required because:

1. **EGL display sharing**: When multiple tests run in parallel threads within
   one process, `eglTerminate` on one thread can tear down a shared EGL display
   while other threads still reference it. This causes intermittent test failures
   that are difficult to reproduce.

2. **G2D driver state**: The `galcore` kernel driver maintains per-process state
   that is not safe to access from concurrent threads creating and destroying
   G2D contexts.

3. **DMA-heap allocation**: Concurrent DMA-heap allocations from multiple test
   threads can exhaust the CMA pool on memory-constrained embedded targets.

This constraint applies to CI (`test.yml`), the Makefile, and local development.
The `cargo nextest` runner already provides per-test process isolation, but
`-j 1` is still specified to prevent DMA/GPU contention across test processes.

### Mask Rendering Benchmarks

The mask rendering benchmarks use a custom in-process harness (`edgefirst-bench`)
instead of Criterion to avoid GPU driver crashes from fork-based benchmarking
on i.MX8/i.MX95 targets.

**Rust benchmark** (`crates/image/benches/mask_benchmark.rs`):
- `decode_masks/proto` — NMS + extract mask coefficients (no mask materialization)
- `decode_masks/materialize` — NMS + extract proto data + materialize pixel masks on CPU
- `draw_masks/{cpu,opengl}` — pre-decoded mask overlay
- `draw_masks_proto/{cpu,opengl}` — fused proto→overlay
- `decode_masks_atlas/{cpu,opengl}` — proto→pixel atlas (all masks in single GPU pass)

Run: `cargo bench -p edgefirst-image --bench mask_benchmark`

**Python benchmarks** (`tests/bench_decode_render.py`):
- `decode() + draw_masks()` — 2-step path (decode to proto-res masks, then draw)
- `draw_masks() [fused]` — single-call fused decode+draw path
- `decode_masks()` — decode masks at output resolution

Run: `python tests/bench_decode_render.py [--iterations N] [--json results.json]`

**Python profiling** (`tests/profile_decode_render.py`):
Isolated hot-loop profiling designed for `perf record`. Separates setup (model
load, EGL init) from the measured loop.

Run: `perf record -F 997 --call-graph dwarf -- python tests/profile_decode_render.py fused`

## Performance Considerations

### Memory Allocation Strategy

The choice of memory type significantly impacts performance depending on the workload:

1. **Heap Memory** (`MemTensor<T>`): Fastest for pure CPU algorithms (image resizing, filtering, format conversion). Standard heap allocation has minimal overhead and is optimized by the OS. Recommended when no hardware acceleration is required.

2. **DMA Memory** (`DmaTensor<T>`): Introduces CPU-level overhead for allocation and memory mapping, but provides substantial benefits when interfacing with hardware accelerators:
   - Zero-copy access from G2D (NXP i.MX graphics processor)
   - Zero-copy access from OpenGL/GPU
   - Zero-copy access from V4L2 video capture and codec engines
   - Hardware DMA operations benefit from DMA-capable memory alignment and page locking
   - Best for workloads that combine CPU processing with hardware acceleration

3. **Shared Memory** (`ShmTensor<T>`): Slowest option with CPU overhead from POSIX shared memory operations. Does not support hardware DMA operations. Use only for cross-process buffer sharing when dma-buf is unavailable due to insufficient permissions, non-Linux platforms, or when persistent memory that survives process termination is required.

**Memory Selection Guidance**:
- Pure CPU workloads (algorithms only): Use `MemTensor` (Heap)
- Hardware-accelerated operations (G2D, OpenGL, V4L2, codec): Use `DmaTensor`
- Cross-process buffer sharing: Use `ShmTensor` (when dma memory cannot be used)

### Image Processing Strategy

The HAL supports multiple image processing backends that are selected automatically based on hardware availability:
- **G2D**: NXP i.MX graphics processor acceleration
- **OpenGL**: GPU-accelerated image processing
- **CPU**: Fallback using vectorized operations and parallelization with Rayon

**BGRA destination format**: BGRA (byte order B, G, R, A) is supported as a
destination-only format for Cairo/Wayland compositing, where the native pixel
format is ARGB32 (big-endian), which is BGRA in memory on little-endian AArch64.
OpenGL renders natively via `GL_BGRA` (`GL_EXT_texture_format_BGRA8888`), G2D
uses the native `G2D_BGRA8888` format, and the CPU backend converts to RGBA
then swizzles R↔B channels in-place.

### C API Performance Recommendations (DMA-BUF / EGL Path)

The following patterns apply when using the DMA-BUF tensor APIs
(`hal_import_image()`, `hal_tensor_from_fd()`)
together with `hal_image_processor_convert()` from C or C++.

> **Reference implementation**: The `bench_preproc` C benchmark
> (`crates/capi/tests/bench_preproc.c`) demonstrates every pattern described
> below in a complete, runnable program. Build it with `make bench` from
> `crates/capi/tests/`. Use it as a starting point for new integrations.

#### Core Principle: Allocate Once, Reuse Every Frame, Free on Exit

Every call to the tensor-from-fd family of functions allocates a new
`BufferIdentity` with a globally unique ID. The OpenGL EGL image cache is
keyed by this ID (see the "EGL Image Cache" section above). A new
`BufferIdentity` means a cache miss, which means a full `eglCreateImageKHR`
import on the next `convert()` call. On the i.MX 8M Plus this costs roughly
0.5--1.5 ms per import — enough to drop below real-time at 30 fps when source
and destination are both re-imported every frame.

The correct lifecycle follows three phases:

```
 INIT                         LOOP                         TEARDOWN
 ──────────────────────────   ──────────────────────────   ─────────────
 Allocate processor           Reuse same tensors           Free tensors
 Allocate src/dst tensors     Call convert()               Free processor
 (from fd or create_image)    (EGL cache hits)
```

#### Phase 1 — Initialization

Allocate all tensors before entering the processing loop. When the source
dimensions are not known until the first frame arrives (e.g., V4L2 resolution
negotiation), allocate on first use and keep the tensors for all subsequent
frames.

> **Important**: When using `hal_image_processor_convert()`, all
> internally-allocated tensors (intermediate buffers, output buffers) **must**
> be created via `hal_image_processor_create_image()`, **not** via
> `hal_tensor_new()` or `hal_tensor_from_fd()`. The processor's
> `create_image` method selects the optimal memory backend for the active
> GPU: DMA-BUF when the GPU uses EGLImage imports (Vivante, Mali), PBO when
> it uses pixel buffer transfers (NVIDIA desktop), and heap as fallback.
> Using `hal_tensor_new()` with a hardcoded memory type bypasses this
> selection and can force a slow transfer path.
>
> **`hal_import_image` is NOT the same as `create_image`.** Despite
> living on the processor object, `hal_import_image()` does not use the
> processor's backend intelligence — it simply wraps an external DMA-BUF
> fd as a DMA tensor. It exists only for importing buffers that are
> **externally allocated** by V4L2, GStreamer, or codec output. Do not
> use it to create intermediate or destination buffers; use
> `hal_image_processor_create_image()` for those.
>
> | Function | Use for | Memory selection |
> |----------|---------|-----------------|
> | `hal_image_processor_create_image()` | Intermediate and output buffers | **Auto** (DMA / PBO / Mem based on GPU) |
> | `hal_import_image()` | External DMA-BUF import only | Always DMA (wraps caller's fd) |
> | `hal_tensor_new()` / `hal_tensor_from_fd()` | Low-level tensor creation | Caller-specified (no GPU awareness) |

```c
// --- Initialization (once) ------------------------------------------------

struct hal_image_processor *proc = hal_image_processor_new();

// Source: external DMA-BUF from a V4L2 capture buffer.
// hal_plane_descriptor_new dups the fd — caller keeps its copy.
struct hal_plane_descriptor *pd = hal_plane_descriptor_new(v4l2_buf.m.fd);
struct hal_tensor *src = hal_import_image(
    proc, pd, NULL, width, height, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
// pd is consumed by hal_import_image — do NOT free

// Intermediate: processor-allocated RGBA for chained conversion.
// MUST use create_image (not hal_tensor_new) for optimal memory backend.
struct hal_tensor *mid = hal_image_processor_create_image(
    proc, model_w, model_h, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);

// Destination: PlanarRgb for model input, also processor-allocated.
// MUST use create_image (not hal_tensor_new) for optimal memory backend.
struct hal_tensor *dst = hal_image_processor_create_image(
    proc, model_w, model_h, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8);
```

When the external buffer has row padding (stride > width * bytes_per_pixel),
set the stride on the plane descriptor:

```c
struct hal_plane_descriptor *pd = hal_plane_descriptor_new(v4l2_buf.m.fd);
hal_plane_descriptor_set_stride(pd, v4l2_fmt.fmt.pix.bytesperline);
struct hal_tensor *src = hal_import_image(
    proc, pd, NULL, width, height, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
// pd is consumed — do NOT free
```

#### Phase 2 — Main Processing Loop

Reuse the same tensor objects on every frame. The EGL image cache hits on
the second and all subsequent iterations because the `BufferIdentity.id` has
not changed.

```c
// --- Main loop (every frame) ----------------------------------------------

while (running) {
    // Upstream writes new pixel data into the DMA-BUF (V4L2 DQBUF, decoder
    // output, etc.). The tensor and its cached EGLImage remain valid — no
    // need to recreate anything. EGLImage is a handle to live physical memory.

    // Two-pass conversion: NV12 → RGBA → PlanarRgb
    hal_image_processor_convert(proc, src, mid,
                                HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
    hal_image_processor_convert(proc, mid, dst,
                                HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);

    // Feed dst to the model ...
}
```

Both `mid` entries (as destination in pass 1 and as source in pass 2) live in
independent EGL image caches (`dst_egl_cache` and `src_egl_cache`), so both
sides achieve cache hits after the first frame.

The `glFinish()` issued at the end of each `convert()` call guarantees
coherency, making chained calls safe without explicit synchronization.

#### Phase 3 — Teardown

Free tensors only when the pipeline is torn down — typically at program exit
or when a pipeline is reconfigured (e.g., resolution change).

```c
// --- Teardown (once) ------------------------------------------------------

hal_tensor_free(dst);
hal_tensor_free(mid);
hal_tensor_free(src);
hal_image_processor_free(proc);
```

#### Buffer Pool Integration (V4L2 / GStreamer)

When upstream provides DMA-BUF fds from a buffer pool (V4L2 MMAP, GStreamer
allocator, codec output ring), a small number of physical buffers are cycled
through a queue. The correct pattern maps each pool slot to a HAL tensor that
is created once and reused whenever that slot is dequeued.

```c
// Pool integration example (V4L2 MMAP with N_BUFS buffers)

#define N_BUFS 4
struct hal_tensor *pool_tensors[N_BUFS] = { NULL };

// At DQBUF time, lazily create or reuse the tensor for this buffer index.
int buf_index = v4l2_buf.index;  // 0..N_BUFS-1

if (pool_tensors[buf_index] == NULL) {
    // First time this pool slot is seen — create the HAL tensor.
    // hal_plane_descriptor_new dups the fd, so V4L2 keeps its copy.
    struct hal_plane_descriptor *pd = hal_plane_descriptor_new(v4l2_buf.m.fd);
    pool_tensors[buf_index] = hal_import_image(
        proc, pd, NULL, width, height,
        HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
    // pd is consumed — do NOT free
}

// Reuse the existing tensor — cache hit on every subsequent dequeue.
hal_image_processor_convert(proc, pool_tensors[buf_index], dst, ...);

// QBUF returns the buffer to the driver; the tensor stays alive.
ioctl(fd, VIDIOC_QBUF, &v4l2_buf);

// --- Teardown (when stopping capture) ---
for (int i = 0; i < N_BUFS; i++) {
    hal_tensor_free(pool_tensors[i]);  // NULL-safe
    pool_tensors[i] = NULL;
}
```

With `hal_tensor_from_fd()` the pattern is similar, but since that function
takes ownership of the fd, the caller must `dup()` first:

```c
// When using hal_tensor_from_fd (takes ownership of fd):
if (pool_tensors[buf_index] == NULL) {
    int duped_fd = dup(v4l2_buf.m.fd);       // keep V4L2's copy alive
    size_t shape[] = { height, width, 1 };    // NV12 luma plane
    pool_tensors[buf_index] = hal_tensor_from_fd(
        HAL_DTYPE_U8, duped_fd, shape, 3, "v4l2_src");
    hal_tensor_set_format(pool_tensors[buf_index], HAL_PIXEL_FORMAT_NV12);
}
```

#### fd Ownership Summary

| Function | fd ownership | When to `dup()` |
|----------|-------------|-----------------|
| `hal_plane_descriptor_new()` | Dups eagerly — caller retains original fd | Never — caller keeps its fd |
| `hal_import_image()` | Consumes both descriptors (success or fail) | Never — descriptors already duped the fd |
| `hal_tensor_from_fd()` | HAL takes ownership | Always, if caller needs the fd afterward |

#### Anti-Patterns

The following patterns cause severe performance regressions. Each is explained
with the underlying cost so that the reason is clear, not just the rule.

**1. Creating a tensor from fd every frame**

```c
// BAD: ~1-3 ms overhead per frame from EGLImage re-import
while (running) {
    struct hal_plane_descriptor *pd = hal_plane_descriptor_new(v4l2_buf.m.fd);
    struct hal_tensor *src = hal_import_image(
        proc, pd, NULL, width, height, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
    hal_image_processor_convert(proc, src, dst, ...);
    hal_tensor_free(src);
}
```

*Why it is slow*: Every call allocates a new `BufferIdentity` with a fresh ID.
The EGL image cache keys on `(BufferIdentity.id, chroma_id)`, so the cache
never hits. Each `convert()` call must call `eglCreateImageKHR` to re-import
the DMA-BUF as a GL texture. On the Vivante GC7000UL this takes 0.5--1.5 ms.
When both source and destination are re-imported, the cost doubles.

**2. Freeing and reallocating tensors between frames**

```c
// BAD: same cost as #1, plus heap allocation overhead
while (running) {
    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, model_w, model_h, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8);
    hal_image_processor_convert(proc, src, dst, ...);
    // ... use dst ...
    hal_tensor_free(dst);
}
```

*Why it is slow*: In addition to EGL image cache misses, every iteration
allocates and frees a DMA-BUF (or PBO), which involves kernel calls
(`dma-heap ioctl` or `glBufferData`). On memory-constrained embedded
targets, CMA pool fragmentation can also cause allocation failures
after sustained operation.

**3. Using the same tensor as both src and dst**

```c
// BAD: undefined behavior
hal_image_processor_convert(proc, tensor, tensor, ...);
```

*Why it fails*: The OpenGL backend binds `src` as a texture and `dst` as a
framebuffer attachment. Sampling from and rendering to the same image in a
single draw call is undefined behavior per the OpenGL ES specification.
Results range from correct output (by accident) to GPU hangs.

**4. Using `hal_tensor_new()` instead of `hal_image_processor_create_image()`**

```c
// BAD: bypasses processor's memory backend selection
size_t shape[] = { 640, 640, 3 };
struct hal_tensor *dst = hal_tensor_new(HAL_DTYPE_U8, shape, 3,
                                         HAL_TENSOR_MEMORY_DMA, "output");
hal_image_processor_convert(proc, src, dst, ...);
```

*Why it is slow*: `hal_image_processor_create_image()` inspects the active GPU
backend and selects the optimal memory type — DMA-BUF for EGLImage-capable
GPUs (Vivante, Mali), PBO for desktop GPUs (NVIDIA), heap for CPU-only.
Calling `hal_tensor_new()` with a hardcoded memory type skips this selection.
On a PBO-preferred system, a DMA tensor forces a slow `glTexSubImage2D`
upload; on a DMA-preferred system, a heap tensor forces a full CPU readback.
Always use `create_image` for tensors that will be passed to `convert()`.

**5. Ignoring row stride for padded buffers**

```c
// BAD: corrupted output (skewed image)
struct hal_plane_descriptor *pd = hal_plane_descriptor_new(padded_fd);
struct hal_tensor *src = hal_import_image(
    proc, pd, NULL, width, height, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
```

*Why it fails*: Many V4L2 drivers and GStreamer allocators pad rows to
alignment boundaries (e.g., 128-byte or 256-byte alignment). If the
stride is not communicated to HAL, the GPU interprets rows at `width *
bpp` spacing while the actual data is at `stride` spacing, producing a
skewed or corrupted image. Set the stride on the plane descriptor via
`hal_plane_descriptor_set_stride()` when
`bytesperline > width * bytes_per_pixel`.

#### Live-Memory Semantics

After a V4L2 decoder, camera ISP, or codec writes new pixel data into a
DMA-BUF, the existing tensor and its cached EGLImage remain valid. Do not
free and recreate the tensor. Simply call `convert()` again. The GPU reads
the updated content because EGLImage is a handle to physical memory, not a
snapshot of its contents at import time. This is the key property that makes
the allocate-once / reuse-every-frame pattern work.

### Decoder Optimization

Decoder implementations use:
- Quantized integer math where applicable
- Vectorized operations via ndarray
- Parallel processing with Rayon
- Early termination in NMS loops

## Thread Safety

All major types implement `Send + Sync`:
- `Tensor<T>`: Safe to share across threads
- `TensorDyn`: Thread-safe
- `ImageProcessor`: Thread-local (create per thread)
- `Decoder`: Thread-safe for read operations

## Error Handling

Consistent error handling throughout:
- Custom `Result<T, Error>` types per crate following the standard Rust pattern
- Each crate defines its own `Error` enum (e.g., `tensor::Error`, `image::Error`, `decoder::Error`)
- Errors are enums with context
- Conversion to PyErr for Python bindings
- All public APIs return `Result<T, Error>` for fallible operations

## Platform Support

| Feature | Linux (i.MX) | Linux (Generic) | macOS | Windows |
|---------|--------------|-----------------|-------|---------|
| DMA Tensors | ✅ | ✅ | ❌ | ❌ |
| PBO Tensors (GPU) | ✅ | ✅ | ❌ | ❌ |
| Shared Memory Tensors | ✅ | ✅ | ✅ | ❌ |
| Heap Tensors | ✅ | ✅ | ✅ | ✅ |
| G2D Acceleration | ✅ | ❌ | ❌ | ❌ |
| OpenGL Acceleration | ✅ (optional) | ✅ (optional) | ❌ | ❌ |
| CPU Fallback | ✅ | ✅ | ✅ | ✅ |

> **Note**: Shared Memory uses POSIX `shm_open` and is available on Unix platforms only (`#[cfg(unix)]`). Windows is not supported for SHM tensors.

## Dependencies

### Key External Dependencies

- **PyO3**: Python bindings
- **ndarray**: N-dimensional arrays
- **rayon**: Data parallelism
- **fast_image_resize**: CPU image operations
- **zune-jpeg/zune-png**: Image decoding
- **dma-heap**: Linux DMA allocation
- **nix**: Unix system calls

### Internal Dependency Graph

```mermaid
graph TD
    EF[edgefirst<br/>top-level re-export]
    Tensor[edgefirst_tensor]
    Image[edgefirst_image]
    Decoder[edgefirst_decoder]
    Tracker[edgefirst_tracker]
    G2D[g2d-sys<br/>optional, Linux only]

    EF --> Tensor
    EF --> Image
    EF --> Decoder
    Image --> Tensor
    Image --> Decoder
    Image -.optional.-> G2D

    Python[edgefirst-hal<br/>Python bindings]
    PyO3[pyo3]
    Numpy[numpy]

    Python --> EF
    Python --> PyO3
    Python --> Numpy

    CAPI[edgefirst-hal-capi<br/>C API bindings]
    CAPI --> EF
    CAPI --> Tracker

    GpuProbe[gpu-probe<br/>binary]
    GpuProbe --> Image

    style EF fill:#fff4e1
    style Python fill:#e1f5ff
    style CAPI fill:#e1f5ff
    style Tracker fill:#e8f5e9
    style GpuProbe fill:#f5f5f5
```

## Source Code Organization

### Repository Structure
```
hal/
├── .github/
│   └── workflows/          # CI/CD automation
│       ├── test.yml        # Rust + Python testing
│       ├── release.yml     # PyPI + crates.io publishing
│       └── nightly.yml     # Nightly builds
├── crates/
│   ├── hal/                # Top-level re-export crate (edgefirst)
│   ├── tensor/             # Zero-copy tensor abstraction
│   ├── image/              # Image processing HAL
│   │   └── src/
│   │       ├── lib.rs      # load_image, save_jpeg, ImageProcessor
│   │       ├── cpu/        # CPU format conversion, resize, masks
│   │       ├── g2d.rs      # NXP G2D hardware accelerator
│   │       ├── gl/         # EGL/OpenGL headless renderer
│   │       │   ├── mod.rs      # Public types, Int8InterpolationMode
│   │       │   ├── processor.rs # GPU rendering pipelines
│   │       │   ├── shaders.rs  # GLSL shader generators
│   │       │   ├── context.rs  # EGL context management
│   │       │   ├── threaded.rs # Thread-safe wrapper
│   │       │   ├── resources.rs # GL resource management
│   │       │   └── cache.rs    # Shader/texture caching
│   │       └── error.rs
│   ├── decoder/            # Model output decoding
│   │   └── src/
│   │       ├── lib.rs      # Public API
│   │       └── decoder/    # Core decode logic
│   │           ├── mod.rs      # Decoder struct, decode methods
│   │           ├── builder.rs  # DecoderBuilder, model type resolution
│   │           ├── configs.rs  # ModelType, DecoderVersion, Nms enums
│   │           ├── postprocess.rs # NMS, dequantization, box parsing
│   │           └── helpers.rs  # Shared utilities
│   ├── tracker/            # Object tracking (ByteTrack)
│   ├── capi/               # C API bindings (cbindgen, staticlib/cdylib)
│   ├── gpu-probe/          # GPU capability probing binary
│   ├── bench/              # Benchmark harness (avoids Criterion fork issues)
│   └── python/             # PyO3 Python bindings
├── tests/                  # Python integration tests
├── testdata/               # Test data (Git LFS)
├── README.md
├── ARCHITECTURE.md         # This document
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── LICENSE                 # Apache-2.0
└── NOTICE                  # Third-party attributions
```

### Crate Dependency Graph
```
python (edgefirst-hal) → edgefirst → tensor, image, decoder
capi → edgefirst → tensor, image, decoder, tracker
image → tensor, decoder (for DetectBox/ProtoData/Segmentation types), g2d-sys
decoder → (standalone — no internal crate deps)
tracker → (standalone — no internal crate deps)
```

### Notable File Sizes

Several modules are significantly larger than typical Rust modules:

| File / Module | Lines | Content |
|---------------|------:|---------|
| `image/src/gl/` (total) | ~8,600 | EGL context, GL processors, shaders, caching, tests |
| `image/src/gl/processor.rs` | ~4,600 | GPU rendering pipelines (convert, masks, atlas) |
| `decoder/src/decoder/` (total) | ~7,000 | Config parsing, decode logic, postprocessing, tests |
| `image/src/lib.rs` | ~5,500 | load_image, save_jpeg, ImageProcessor |
| `capi/src/decoder.rs` | ~2,800 | C API decoder bindings |

---

## Appendix A: Multi-Plane DMA-BUF Limitation

### Current State

The HAL's DMA-BUF integration assumes a **single file descriptor per buffer**
with all planes stored contiguously. This is baked into multiple layers:

| Layer | Assumption | Code Location |
|-------|-----------|---------------|
| `DmaTensor<T>` | Single `fd: OwnedFd` field | `crates/tensor/src/dma.rs:31` |
| `TensorTrait::from_fd()` | Takes one `OwnedFd` | `crates/tensor/src/dma.rs:88` |
| `hal_tensor_from_fd()` C API | Takes one `int fd` | `crates/capi/include/edgefirst/hal.h` |
| `TensorDyn` NV12 shape | `[H*3/2, W]` — contiguous Y+UV | `crates/tensor/src/lib.rs` |
| EGL NV12 import | Same fd for both planes, UV offset = `W*H` | `crates/image/src/opengl_headless.rs:4492–4501` |

This works correctly when the kernel allocates NV12 from a single CMA/system
DMA-heap buffer (e.g. `hal_tensor_new()`, `hal_image_processor_create_image()`).
The Y and UV planes are contiguous in physical memory and share one fd.

### The Problem: Multi-Planar Formats

Video hardware on NXP i.MX platforms frequently produces **multi-planar**
NV12 buffers where Y and UV reside in separate DMA-BUF allocations, each
with its own file descriptor:

| Source | V4L2 Format | Planes | Behavior |
|--------|-------------|--------|----------|
| VPU (Hantro/Amphion) via `v4l2h264dec` | `V4L2_PIX_FMT_NV12M` (NM12) | 2 fds | Y and UV in separate DMA-BUFs |
| NeoISP via `libcamerasrc` | `V4L2_PIX_FMT_NV12M` (NM12) | 2 fds | Y and UV in separate DMA-BUFs |
| MIPI-CSI direct capture | `V4L2_PIX_FMT_NV12` (NV12) | 1 fd | Contiguous — works today |

When GStreamer negotiates `video/x-raw(memory:DMABuf), format=DMA_DRM,
drm-format=NV12`, the upstream element may deliver buffers with 2 `GstMemory`
blocks (one per plane).

### Multi-Plane Support (Implemented)

The HAL supports multi-plane DMA-BUF NV12/NV16 via a two-tensor approach
rather than extending `DmaTensor` with multiple fds:

**C API**: `hal_import_image(proc, y_pd, uv_pd, width, height, format, dtype)`
takes separate Y and UV plane descriptors, wraps each into its own
`Tensor<u8>` via `from_fd()`, and combines them with `Tensor::from_planes()`.
Per-plane stride and offset can be set on each descriptor before import.

**Tensor crate**: `Tensor::from_planes(luma, chroma, PixelFormat::Nv12)` stores the
two tensors as separate planes inside the `Tensor`, preserving their
independent DMA-BUF allocations for zero-copy GPU import.

**OpenGL path**: `create_image_from_dma2()` uses per-plane fds in EGL
attributes (`DMA_BUF_PLANE0_FD → y_fd`, `DMA_BUF_PLANE1_FD → uv_fd`),
each at offset 0.

| Scenario | Zero-Copy | Notes |
|----------|-----------|-------|
| Single-fd NV12 DMA-BUF | Yes | V4L2 single-planar capture, HAL-allocated buffers |
| Single-fd YUYV/RGB/RGBA DMA-BUF | Yes | Always single-plane |
| System memory input | N/A | Copied into HAL tensor regardless |
| Multi-fd NV12/NV16 DMA-BUF | Yes | Via `hal_import_image()` with two plane descriptors |

### GStreamer Integration (External)

The `edgefirstcameraadaptor` element detects multi-plane buffers
(`gst_buffer_n_memory() > 1`) and extracts per-plane fds via
`gst_dmabuf_memory_get_fd()` on each `GstMemory` block, passing them to
`hal_import_image()` via separate plane descriptors for zero-copy import.

### Tracking

This work was implemented under **EDGEAI-1107**.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development environment setup
- Build instructions
- Testing guidelines
- Code style standards
- Pull request process

## Support

For questions, issues, or contributions:
- **Documentation**: https://doc.edgefirst.ai
- **GitHub Issues**: https://github.com/EdgeFirstAI/hal/issues
- **Email**: support@au-zone.com
